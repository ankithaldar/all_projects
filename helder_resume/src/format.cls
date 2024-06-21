% ******************************************************************************
% Created by
% Ankit Haldar
% ******************************************************************************



% ******************************************************************************
% CHANGELOG
%
% ******************************************************************************



% ******************************************************************************
% INTRO OPTIONS
\ProvidesClass{ats_friendly_template}[2024/04/26 CV class]
\NeedsTeXFormat{LaTeX2e}

\DeclareOption{print}{\def\@cv@print{}}
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{article}
}
\ProcessOptions\relax

\LoadClass{article}
% ******************************************************************************



% ******************************************************************************
% PACKAGES
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage[usenames,dvipsnames]{color}
\usepackage{enumitem}
\input{glyphtounicode}
\usepackage{ragged2e}
% \usepackage[T1]{fontenc}

% Ensure that generate pdf is machine readable/ATS parsable
\pdfgentounicode=1
% ******************************************************************************


% ******************************************************************************
% SET PAGE STYLE
\pagestyle{fancy}
\fancyhf{} % clear all header and footer fields
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Adjust margins
\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-0.5in}
\addtolength{\textheight}{1.0in}
\setlength{\footskip}{3.60004pt}

% ******************************************************************************



% ******************************************************************************
% DEFINE COLORS
% ******************************************************************************



% ******************************************************************************
% DEFINE FONTS
% ******************************************************************************



% ******************************************************************************
% DEFINE COMMANDS

\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}


% Name Command
\newcommand{\cvname}[2]{
  \begin{center}
    \textbf{\Huge {#1} {#2}} \\ \vspace{-10pt}
  \end{center}
}


% Preferred Position Command
\newcommand{\cvposition}[1]{
  \begin{center}
    \large {#1} \\ \vspace{-10pt}
  \end{center}
}


% Other details like phone number, email etc
\newcommand{\cvdetails}[4]{
  \begin{center}
    \normalsize {#1}
    $|$
    \href{mailto:#2}{\underline{#2}}
    $|$
    \href{https://#3}{\underline{#3}}
    $|$
    {#4}
  \end{center}
}



\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\Large
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]



\newcommand{\resumeItem}[1]{
  \item\small{
    {#1 \vspace{-1pt}}
  }
}

\newcommand{\resumeSubheading}[4]{
  \vspace{0pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubSubheading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \textit{\small#1} & \textit{\small #2} \\
    \end{tabular*}\vspace{-5pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{0.5pt}}

\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}
% ******************************************************************************
