\ProvidesClass{twentysecondcv}[2017/01/08 CV class]
\LoadClass{article}
\NeedsTeXFormat{LaTeX2e}

%----------------------------------------------------------------------------------------
%  REQUIRED PACKAGES
%----------------------------------------------------------------------------------------

\RequirePackage[sfdefault]{ClearSans}
\RequirePackage[T1]{fontenc}
\RequirePackage{tikz}
\RequirePackage{xcolor}
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{ragged2e}
\RequirePackage{etoolbox}
\RequirePackage{ifmtarg}
\RequirePackage{ifthen}
\RequirePackage{pgffor}
\RequirePackage{marvosym}
\RequirePackage{parskip}
\RequirePackage{enumitem}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

%----------------------------------------------------------------------------------------
%  COLOURS
%----------------------------------------------------------------------------------------

\definecolor{white}{RGB}{255,255,255}
\definecolor{gray}{HTML}{4D4D4D}
\definecolor{sidecolor}{HTML}{E7E7E7}
\definecolor{mainblue}{HTML}{0E5484}
\definecolor{maingray}{HTML}{B9B9B9}
\definecolor{sectionblue}{HTML}{073B4C}
\definecolor{bfcolor}{HTML}{023047}

%----------------------------------------------------------------------------------------
%  MISC CONFIGURATIONS
%----------------------------------------------------------------------------------------

% \renewcommand{\bfseries}{\color{bfcolor}{\textbf{#1}}} % Make \textbf produce coloured text instead

\newcommand\boldcolor[1]{\textbf{\color{bfcolor}{#1}}}

\pagestyle{empty} % Disable headers and footers

\setlength{\parindent}{0pt} % Disable paragraph indentation

%----------------------------------------------------------------------------------------
%  SIDEBAR DEFINITIONS
%----------------------------------------------------------------------------------------

\setlength{\TPHorizModule}{1cm} % Left margin
\setlength{\TPVertModule}{1cm} % Top margin

\newlength\imagewidth
\newlength\imagescale
\pgfmathsetlength{\imagewidth}{5cm}
\pgfmathsetlength{\imagescale}{\imagewidth/600}

\newlength{\TotalSectionLength} % Define a new length to hold the remaining line width after the section title is printed
\newlength{\SectionTitleLength} % Define a new length to hold the width of the section title
\newcommand{\profilesection}[1]{%
  \setlength\TotalSectionLength{\linewidth}% Set the total line width
  \settowidth{\SectionTitleLength}{\Large #1 }% Calculate the width of the section title
  \addtolength\TotalSectionLength{-\SectionTitleLength}% Subtract the section title width from the total width
  \addtolength\TotalSectionLength{-2.22221pt}% Modifier to remove overfull box warning
  \vspace{8pt}% Whitespace before the section title
  {\color{black!100} \Large #1 \rule[0.15\baselineskip]{\TotalSectionLength}{1pt}}% Print the title and auto-width rule
}

% Define custom commands for CV info
\newcommand{\cvdate}[1]{\renewcommand{\cvdate}{#1}}
\newcommand{\cvmail}[1]{\renewcommand{\cvmail}{#1}}
\newcommand{\cvnumberphone}[1]{\renewcommand{\cvnumberphone}{#1}}
\newcommand{\cvsite}[1]{\renewcommand{\cvsite}{#1}}
\newcommand{\cvlocation}[1]{\renewcommand{\cvlocation}{#1}}
\newcommand{\cvaddress}[1]{\renewcommand{\cvaddress}{#1}}
\newcommand{\education}[1]{\renewcommand{\education}{#1}}
\newcommand{\skills}[1]{\renewcommand{\skills}{#1}}
\newcommand{\orginitiatives}[1]{\renewcommand{\orginitiatives}{#1}}
\newcommand{\interests}[1]{\renewcommand{\interests}{#1}}
\newcommand{\awards}[1]{\renewcommand{\awards}{#1}}
\newcommand{\hobbies}[1]{\renewcommand{\hobbies}{#1}}
\newcommand{\languages}[1]{\renewcommand{\languages}{#1}}



\newcommand{\cvname}[1]{\renewcommand{\cvname}{#1}}
\newcommand{\cvjobtitle}[1]{\renewcommand{\cvjobtitle}{#1}}

% Command for printing the contact information icons
\newcommand*\icon[1]{\tikz[baseline=(char.base)]{\node[shape=circle,draw,inner sep=1pt, fill=mainblue,mainblue,text=white] (char) {#1};}}

% Command for printing skill progress bars


% Command for printing skills text
\newcommand\skillstext[1]{
  \renewcommand{\skillstext}{
    \begin{flushleft}
      \foreach [count=\i] \x/\y in {#1}{
        \x$ \star $\y
      }
    \end{flushleft}
  }
}

%----------------------------------------------------------------------------------------
%  SIDEBAR LAYOUT
%----------------------------------------------------------------------------------------

\newcommand{\makeprofile}{
  \begin{tikzpicture}[remember picture,overlay]
      \node [rectangle, fill=sidecolor, anchor=north, minimum width=9cm, minimum height=\paperheight+1cm] (box) at (-5cm,0.5cm){};
  \end{tikzpicture}

  %------------------------------------------------

  \begin{textblock}{6}(0.5, 0.2)

    %------------------------------------------------

    \ifthenelse{\equal{\profilepic}{}}{}{
      \begin{center}
        \begin{tikzpicture}[x=\imagescale,y=-\imagescale]
          \clip (600/2, 567/2) circle (567/2);
          \node[anchor=north west, inner sep=0pt, outer sep=0pt] at (0,0) {\includegraphics[width=\imagewidth]{\profilepic}};
        \end{tikzpicture}
      \end{center}
    }

    %------------------------------------------------

    {\Huge\color{mainblue}\cvname}

    %------------------------------------------------

    {\LARGE\color{black!80}\cvjobtitle}

    %------------------------------------------------

    \renewcommand{\arraystretch}{1.6}
    \begin{tabular}{p{0.5cm} @{\hskip 0.5cm}p{5cm}}
      \ifthenelse{\equal{\cvdate}{}}{}{\textsc{\Large\icon{\Info}} & \cvdate\\}
      \ifthenelse{\equal{\cvmail}{}}{}{\textsc{\large\icon{@}} & \href{mailto:\cvmail}{\cvmail}\\
      \ifthenelse{\equal{\cvnumberphone}{}}{}{\textsc{\Large\icon{\Telefon}} & \cvnumberphone\\}
      \ifthenelse{\equal{\cvsite}{}}{}{\textsc{\large\textbf{\icon{in}}} & \href{https://www.linkedin.com/in/\cvsite}{\cvsite}\\}
      \ifthenelse{\equal{\cvaddress}{}}{}{\textsc{\Large\icon{\Letter}} & \cvaddress}}
    \end{tabular}

    %------------------------------------------------

    \ifthenelse{\equal{\education}{}}{}{
      \profilesection{Education}
      \begin{flushleft}
        \education
      \end{flushleft}
    }

    %------------------------------------------------
    \ifthenelse{\equal{\skills}{}}{}{
      \profilesection{Skills}
      \begin{flushleft}
        \skills
      \end{flushleft}
    }

    %------------------------------------------------

    \ifthenelse{\equal{\orginitiatives}{}}{}{
      \profilesection{Org-Initiatives}
      \begin{flushleft}
        \orginitiatives
      \end{flushleft}
    }

    %------------------------------------------------

    \ifthenelse{\equal{\interests}{}}{}{
      \profilesection{Interests}
      \begin{flushleft}
        \interests
      \end{flushleft}
    }

    %------------------------------------------------

    \ifthenelse{\equal{\awards}{}}{}{
      \profilesection{Awards}
      \begin{flushleft}
        \awards
      \end{flushleft}
    }

    %------------------------------------------------

    \ifthenelse{\equal{\hobbies}{}}{}{
      \profilesection{Hobbies}
      \begin{flushleft}
        \hobbies
      \end{flushleft}
    }

    %------------------------------------------------

    \ifthenelse{\equal{\languages}{}}{}{
      \profilesection{Languages}
      \begin{flushleft}
        \languages
      \end{flushleft}
    }

    %------------------------------------------------

  \end{textblock}
}

%----------------------------------------------------------------------------------------
%  COLOURED SECTION TITLE BOX
%----------------------------------------------------------------------------------------

% Command to create the rounded boxes around the first three letters of section titles
\newcommand*\round[2]{%
  \tikz[baseline=(char.base)]\node[anchor=north west, draw,rectangle, rounded corners, inner sep=1.6pt, minimum size=5.5mm, text height=3.6mm, fill=#2,#2,text=white](char){#1};%
}

\newcounter{colorCounter}
\newcommand{\sectioncolor}[1]{%
  {%
    \round{#1}{
      \ifcase\value{colorCounter}%
      sectionblue\or%
      mainblue\or%
      maingray\or%
      mainblue\or%
      maingray\or%
      mainblue\or%
      maingray\or%
      mainblue\or%
      maingray\or%
      mainblue\else%
      maingray\fi%
    }%
  }%
  \stepcounter{colorCounter}%
}

\renewcommand{\section}[1]{
  {%
    \color{gray}%
    \Large\sectioncolor{#1}%
  }
}

\renewcommand{\subsection}[1]{
  \par\vspace{.5\parskip}{%
    \large\color{gray} #1%
  }
  \par\vspace{.25\parskip}%
}

%----------------------------------------------------------------------------------------
%  LONG LIST ENVIRONMENT
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}

% New environment for the long list
\newenvironment{twenty}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}

\newcommand{\twentyitem}[4]{%
  #1&\parbox[t]{0.86\textwidth}{%
    \textbf{\color{bfcolor}{\large#2}}%
    \hfill%
    {\small#3}\\%
    #4\vspace{\parsep}%
  }\\
}

%----------------------------------------------------------------------------------------
%  SMALL LIST ENVIRONMENT
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}

% New environment for the small list
\newenvironment{twentyshort}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}

\newcommand{\twentyitemshort}[2]{%
  #1&\parbox[t]{0.83\textwidth}{%
    \textbf{#2}%
  }\\
}

%----------------------------------------------------------------------------------------
%  MARGINS AND LINKS
%----------------------------------------------------------------------------------------

\RequirePackage[left=7.6cm,top=0.1cm,right=1cm,bottom=0.2cm,nohead,nofoot]{geometry}

\RequirePackage{hyperref}
