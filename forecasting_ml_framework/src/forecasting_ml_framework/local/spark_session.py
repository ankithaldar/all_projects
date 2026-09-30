#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A local Spark session, and the two things that make one work here.

The framework builds its Spark session through the single platform builder, and
that builder is written for a hosted cluster. Running it on a workstation needs
two adjustments, and both are environmental rather than behavioural -- the
framework's own session-building logic is not changed or bypassed.

**The Java version.** Spark 3.5 runs on Java 8, 11 and 17. A newer JDK fails at
startup, not with a clear message: ``UserGroupInformation.getCurrentUser`` calls
``Subject.getSubject``, which was removed in JDK 24, so the driver dies inside
its own initialisation with a stack trace that names none of the three
technologies involved. A local run therefore points ``JAVA_HOME`` at a Java 17
runtime before the session is created.

**The worker's Python.** A Spark executor launches a *separate* Python process
and picks one from the environment, not from the virtualenv the driver came
from. When the two minor versions differ, PySpark refuses to run and says so,
but only after the driver has already started. Setting ``PYSPARK_PYTHON`` to the
interpreter running the driver removes the mismatch at its source.

Both are set through the process environment before the session is built, so
nothing in the framework needs to know it is running locally.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Environment variable naming a supported Java runtime for the local session.
JAVA_HOME_VARIABLE = 'FORECASTING_ML_JAVA_HOME'

#: Java majors the pinned Spark can run on. Spark has never supported a JVM beyond
#: 17, and the failure it reports for a newer one arrives from the Hadoop security
#: layer as ``UnsupportedOperationException: getSubject is not supported``, naming
#: neither Java nor the version constraint that was violated.
SUPPORTED_JAVA_MAJORS = (8, 11, 17)


def configure_local_environment() -> dict[str, str]:
  """Point Spark at a usable Java runtime and at the driver's own Python.

  Raises:
    LocalEnvironmentError: If no Java runtime the pinned Spark version supports can
      be located. Failing here is the point: Spark's own failure for an unsupported
      JVM arrives from deep inside the Hadoop security layer as
      ``UnsupportedOperationException: getSubject is not supported``, which names
      neither Java nor Spark and reads as a Spark bug rather than as a version
      mismatch on the host.
  """
  java_home = _resolve_java_home()
  if java_home:
    os.environ['JAVA_HOME'] = java_home
    _prepend_path(os.path.join(java_home, 'bin'))

  # The worker must be the same interpreter as the driver. Resolving it from
  # sys.executable rather than a name means it is correct whether the run was
  # started with `.venv/bin/python`, `python` from an activated environment, or
  # an absolute path.
  import sys  # noqa: PLC0415

  os.environ['PYSPARK_PYTHON'] = sys.executable
  os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

  # A workstation has no routable hostname in the usual sense, and Spark's
  # default bind can fail to resolve it, which surfaces as an obscure
  # NetException rather than as a configuration problem.
  os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
  os.environ.setdefault('SPARK_HOME', _spark_home() or '')
  if not os.environ.get('SPARK_HOME'):
    os.environ.pop('SPARK_HOME', None)

  return {
    'JAVA_HOME': os.environ.get('JAVA_HOME', '') or '(unset)',
    'PYSPARK_PYTHON': os.environ['PYSPARK_PYTHON'],
    'SPARK_LOCAL_IP': os.environ['SPARK_LOCAL_IP'],
  }


def _resolve_java_home() -> str:
  """Locate a Java runtime the pinned Spark version supports.

  The explicit variable wins, because a machine with several JDKs installed needs
  to say which one. Otherwise a supported runtime is searched for among the usual
  locations, and an already-correct ``JAVA_HOME`` is left alone.

  Returns:
    The runtime directory.

  Raises:
    LocalEnvironmentError: If the only runtime on the machine is a version the
      pinned Spark cannot run on. Returning ``None`` in that case is what produced
      the deeply unhelpful failure this now replaces: Spark reported
      ``UnsupportedOperationException: getSubject is not supported`` from inside
      the Hadoop security layer, which names neither Java nor the version
      constraint it had violated.
  """
  override = os.getenv(JAVA_HOME_VARIABLE)
  if override:
    if not Path(override, 'bin', 'java').exists():
      raise LocalEnvironmentError(
        f'{JAVA_HOME_VARIABLE} points at {override!r}, which has no bin/java.'
      )
    return override

  current = os.getenv('JAVA_HOME', '')
  if current and _java_major(current) in SUPPORTED_JAVA_MAJORS:
    return current

  found = []
  for candidate in sorted(Path('/usr/lib/jvm').glob('*')) if Path('/usr/lib/jvm').is_dir() else []:
    major = _java_major(str(candidate))
    if major in SUPPORTED_JAVA_MAJORS:
      return str(candidate)
    found.append(f'{candidate.name} (Java {major if major else "unknown"})')

  raise LocalEnvironmentError(
    f'No Java runtime Spark can use was found. Supported majors: {sorted(SUPPORTED_JAVA_MAJORS)}. '
    f'Candidates seen under /usr/lib/jvm: {found or ["none"]}. Install a supported JDK, or set '
    f'{JAVA_HOME_VARIABLE} to one. Spark reports an unsupported JVM as an opaque security-layer '
    'exception, so this check is the one that names the actual problem.',
    supported=list(SUPPORTED_JAVA_MAJORS),
    candidates=found,
  )


def _java_major(java_home: str) -> int | None:
  """Return the major version of a Java runtime.

  Args:
    java_home: The runtime directory.

  Returns:
    The major version, or ``None`` when it cannot be determined.
  """
  import subprocess  # noqa: PLC0415

  binary = Path(java_home, 'bin', 'java')
  if not binary.exists():
    return None
  try:
    output = subprocess.run(  # noqa: S603 - a fixed, resolved path
      [str(binary), '-version'], capture_output=True, text=True, timeout=30, check=False
    )
  except (OSError, subprocess.SubprocessError):
    return None
  for line in (output.stderr or output.stdout or '').splitlines():
    if 'version' not in line or '"' not in line:
      continue
    major = _parse_java_version(line.split('"')[1])
    if major is not None:
      return major
  return None


def _parse_java_version(token: str) -> int | None:
  """Extract the major version from a ``java -version`` version token.

  The legacy scheme is the whole reason this is a function. Java 8 and earlier
  report a two-component version whose *first* component is always ``1``, so
  ``"1.8.0_402"`` is Java 8. Reading the first component therefore reports
  version **1**, and 1 is not in the supported set -- so a perfectly good Java 8
  was rejected exactly as reliably as an unsupported one, and the auto-detection
  could never succeed on the oldest runtime it was written to support. From Java
  9 the scheme changed and the first component *is* the major version.

  Splitting on the first dot is not the alternative: that is what produced
  ``1709`` from ``"17.0.9"`` and ``1804`` from ``"1.8.0_402"``, and neither
  equals a supported major either.

  Args:
    token: The quoted version string, e.g. ``'1.8.0_402'`` or ``'17.0.9'``.

  Returns:
    The major version, or ``None`` when the token is not numeric.
  """
  match = re.match(r'(\d+)(?:\.(\d+))?', token.strip())
  if not match:
    return None
  first = int(match.group(1))
  second = match.group(2)
  # The legacy scheme: a leading 1 is the compatibility marker, and the real
  # major version is the component after it.
  if first == 1 and second is not None:
    return int(second)
  return first


def _spark_home() -> str | None:
  """Locate the Spark distribution the installed ``pyspark`` points at.

  Returns:
    The directory, or ``None`` when it cannot be determined.
  """
  try:
    import pyspark  # noqa: PLC0415
  except ImportError:
    return None
  return str(Path(pyspark.__file__).resolve().parent)


def _prepend_path(directory: str) -> None:
  """Put a directory at the front of ``PATH``.

  Args:
    directory: The directory to prepend.
  """
  current = os.environ.get('PATH', '')
  parts = [part for part in current.split(os.pathsep) if part and part != directory]
  os.environ['PATH'] = os.pathsep.join([directory, *parts])


def create_local_session(app_name: str = 'forecasting-ml-local', partitions: int = 4) -> Any:
  """Create the local Spark session the framework's pipeline nodes require.

  The tuning is deliberately minimal. Every setting here has a production
  counterpart in the Spark tuning document, and the point of this session is to
  be *identical in behaviour* to the hosted one while running on a workstation --
  so the only differences are the ones that cannot be helped: no cluster manager,
  no dynamic allocation, and the UI off.

  Args:
    app_name: The application name shown in the Spark log.
    partitions: The shuffle partition count. A workstation has few cores, so the
      production value would create far more tasks than can run concurrently.

  Returns:
    An active ``SparkSession``.
  """
  from pyspark.sql import SparkSession  # noqa: PLC0415

  session = (
    SparkSession.builder.master('local[2]')
    .appName(app_name)
    .config('spark.ui.enabled', 'false')
    .config('spark.sql.shuffle.partitions', str(partitions))
    .config('spark.sql.adaptive.enabled', 'true')
    .config('spark.driver.host', '127.0.0.1')
    .config('spark.driver.bindAddress', '127.0.0.1')
    .config('spark.sql.session.timeZone', 'UTC')
    .getOrCreate()
  )
  session.sparkContext.setLogLevel('ERROR')
  LOGGER.info('Started a local Spark session', extra={'version': session.version, 'app': app_name})
  return session


class LocalEnvironmentError(RuntimeError):
  """Raised when the local runtime cannot be configured for Spark.

  Carries the same structured ``**context`` contract as the framework's own
  exception hierarchy, so the details a caller needs -- which Java versions are
  supported, which were found -- are attached rather than interpolated into the
  message alone.
  """

  def __init__(self, message: str, **context: object) -> None:
    """Build the error.

    Args:
      message: The failure description.
      context: Structured detail attached to the exception.
    """
    super().__init__(message)
    self.message = message
    self.context = dict(context)
