%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------
% Rho
% LaTeX Class
% Version 2.3.1 (28/04/2024)
%
% Author: 
% Guillermo Jimenez (memo.notess1@gmail.com)
% --------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------
% You may modify 'rho.cls' file according to your
% needs and preferences. This LaTeX class file defines 
% the document layout and behavior.
% --------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------
% 						 WARNING!
% --------------------------------------------------------
% It is important to proceed with caution and ensure that 
% any modifications made are compatible with LaTeX 
% syntax and conventions to avoid errors or unexpected 
% behavior in the document compilation process.
% --------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------
% CLASS CONFIGURATION
%----------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rho}[2024/04/28 Rho LaTeX class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions\relax
\LoadClass[twocolumn]{extarticle}
\AtEndOfClass{\RequirePackage{microtype}}

%----------------------------------------------------------
% REQUIRED PACKAGES
%----------------------------------------------------------

\RequirePackage[utf8]{inputenc}
\RequirePackage{etoolbox}
\RequirePackage[framemethod=tikz]{mdframed}
\RequirePackage{titling}
\RequirePackage{lettrine}
\RequirePackage{microtype}
\RequirePackage[bottom,hang,flushmargin,ragged]{footmisc}
\RequirePackage{fancyhdr}
\RequirePackage{xifthen}
\RequirePackage{adjustbox}
\RequirePackage{adforn}
\RequirePackage{lastpage}
\RequirePackage[explicit]{titlesec}
\RequirePackage{booktabs}
\RequirePackage{array} 
\RequirePackage{caption}
\RequirePackage{setspace}
\RequirePackage{iflang}
\RequirePackage{listings}
\RequirePackage{lipsum}
\RequirePackage{fontawesome5}    % For icons
\RequirePackage{chemfig}         % Chemical structures
\RequirePackage{circuitikz}      % Circuits schematics
\RequirePackage[switch]{lineno}
\RequirePackage{supertabular}
\RequirePackage{matlab-prettifier}
\RequirePackage{listings}
\RequirePackage{csquotes}
\RequirePackage{ragged2e}
\RequirePackage{ccicons}
\RequirePackage{imakeidx}
% \RequirePackage{pbalance}

%----------------------------------------------------------
% GEOMETRY PACKAGE SETUP
%----------------------------------------------------------

\RequirePackage[
    left=1.25cm,
    right=1.25cm,
    top=2cm,
    bottom=2cm,
    headsep=0.75cm,
    %showframe
]{geometry}
\setlength{\columnsep}{15pt}
% \renewcommand{\baselinestretch}{1}    % To change the line spacing

%----------------------------------------------------------
% PACKAGES FOR FIGURES
%----------------------------------------------------------

\usepackage{graphicx}
\RequirePackage{here}

%----------------------------------------------------------
% PACKAGES FOR TABLES
%----------------------------------------------------------

\usepackage{adjustbox}
\RequirePackage{colortbl}
\RequirePackage{tcolorbox}

%----------------------------------------------------------
% MATH PACKAGES
%----------------------------------------------------------

%!TEX In case of using another font that is not stix2 uncomment 'amssymb'

\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{mathtools}
% \RequirePackage{amssymb}

% Equation skip value
\newlength{\eqskip}\setlength{\eqskip}{8pt}
\expandafter\def\expandafter\normalsize\expandafter{%
    \normalsize%
    \setlength\abovedisplayskip{\eqskip}%
    \setlength\belowdisplayskip{\eqskip}%
    \setlength\abovedisplayshortskip{\eqskip-\baselineskip}%
    \setlength\belowdisplayshortskip{\eqskip}%
}

%----------------------------------------------------------
% FONTS
%----------------------------------------------------------

\usepackage[notextcomp]{stix2}
\RequirePackage[scaled]{helvet}
\renewcommand{\ttdefault}{lmtt}

%----------------------------------------------------------

% \RequirePackage{pifont}
% \usepackage[sc]{mathpazo}	% Use the Palatino font
% \usepackage[oldstyle,oldstylemath]{kpfonts}	% Stylish old font
% \RequirePackage{times}
% \RequirePackage{XCharter}
% \RequirePackage{mathpazo} 
% \RequirePackage{lmodern}

%----------------------------------------------------------
% URLs STYLE
%----------------------------------------------------------

\RequirePackage{url}
\RequirePackage{xurl}
\renewcommand\UrlFont{\selectfont}

%----------------------------------------------------------

\RequirePackage[colorlinks=true, allcolors=rhocolor]{hyperref}
\RequirePackage{cleveref}
\RequirePackage{bookmark}

%----------------------------------------------------------
% ITEMS
%----------------------------------------------------------

\RequirePackage{enumitem}
\setlist{noitemsep}

%----------------------------------------------------------
% GRAPHICS PATH 
%----------------------------------------------------------

\setlength{\columnsep}{15pt}
\graphicspath{{Figures/}{./}} 

%----------------------------------------------------------
% FIRST PAGE, HEADER AND FOOTER
%----------------------------------------------------------

% Commands
\newcommand{\institution}[1]{\def\@institution{#1}}
\newcommand{\etal}[1]{\def\@etal{#1}}
\newcommand{\footinfo}[1]{\def\@footinfo{#1}}
\newcommand{\smalltitle}[1]{\def\@smalltitle{#1}}

\pagestyle{fancy}
    \pagenumbering{arabic}

% First page style
\fancypagestyle{firststyle}{
    \cfoot{}%
    \lhead{}%
    \fancyfoot[L]{\footnotesize\sffamily\@footinfo}
    \fancyfoot[R]{
    	\footnotesize\sffamily\@institution\hspace{10pt}{\bfseries\@date}\hspace{10pt}\@smalltitle\hspace{10pt}\thepage-\pageref{LastPage}
	} 
}

% Header
\fancyhead[C]{}
\fancyhead[RE]{\footnotesize\sffamily{\itshape\@title}\hspace{10pt}{\bfseries\thepage}}
\fancyhead[LO]{\footnotesize\sffamily{\bfseries\thepage}\hspace{10pt}{\itshape\@title}}
\fancyhead[LE]{}
\fancyhead[RO]{}

% Footer
\fancyfoot[C]{}
\fancyfoot[RO]{\footnotesize\sffamily\@institution\hspace{10pt}{\bfseries\@date}\hspace{10pt}\@smalltitle}
\fancyfoot[LO]{\footnotesize\sffamily\itshape\@etal}
\fancyfoot[RE]{\footnotesize\sffamily\itshape\@etal}
\fancyfoot[LE]{\footnotesize\sffamily\@footinfo}

\renewcommand{\headrulewidth}{0pt} % No header rule
\renewcommand{\footrulewidth}{0pt} % No footer rule

%----------------------------------------------------------
% RHO START ~ LETTRINE
%----------------------------------------------------------

\newcommand{\dropcapfont}{\color{rhocolor}\bfseries\fontsize{25pt}{28pt}\selectfont}
\newcommand{\rhostart}[1]{\lettrine[lines=2,lraise=0,findent=2pt, nindent=0em]{{\dropcapfont{#1}}}{}}

%----------------------------------------------------------
% CORRESPONDING AUTHOR AND ARTICLE INFORMATION 
%----------------------------------------------------------

\newcommand{\rhoinfofont}{\normalfont\sffamily\linespread{1}\fontsize{7.8}{9}\selectfont}
\newcommand{\received}[1]{\def\@received{#1}}
\newcommand{\revised}[1]{\def\@revised{#1}}
\newcommand{\accepted}[1]{\def\@accepted{#1}}
\newcommand{\published}[1]{\def\@published{#1}}
\newcommand{\corres}[1]{\def\@corres{#1}}
\newcommand{\email}[1]{\def\@email{#1}}
\newcommand{\doi}[1]{\def\@doi{#1}}
\newcommand{\license}[1]{\def\@license{#1}}

% Article information language (spanish and english). Change the second {} if using another language babel

\newcommand{\correslanguage}{
    \iflanguage{spanish}{
        {Autor de correspondencia: }%
    }{%
        {Corresponding author: }%
    }%
}

\newcommand{\emaillanguage}{
    \iflanguage{spanish}{
        {Correo electr\'onico: }%
    }{%
        {E-mail address: }%
    }%
}

\newcommand{\receivedlanguage}{
    \iflanguage{spanish}{
        {Recibido: }%
    }{%
        {Received: }%
    }%
}

\newcommand{\revisedlanguage}{
    \iflanguage{spanish}{
        {Revisado: }%
    }{%
        {Revised: }%
    }%
}

\newcommand{\acceptedlanguage}{
    \iflanguage{spanish}{
        {Aceptado: }%
    }{%
        {Accepted: }%
    }%
}

\newcommand{\publishedlanguage}{
    \iflanguage{spanish}{
        {Publicado: }%
    }{%
        {Published: }%
    }%
}


\newcommand{\rhoinfo}{
    \rhoinfofont
    \hrule
        \vskip6.5pt
            {\ifx\@corres\undefined\else\textcolor{rhocolor}{\bfseries\ignorespaces\correslanguage}\@corres\fi} 
            {\ifx\@email\undefined\else\textcolor{rhocolor}{\itshape\ignorespaces\emaillanguage}\@email\fi\par}
            {\ifx\@doi\undefined\else\textcolor{rhocolor}{\bfseries DOI: }\@doi\fi\par}
            {\ifx\@received\undefined\else\textcolor{rhocolor}{\bfseries\ignorespaces\receivedlanguage}\@received\fi}
                \hspace{10pt}
            {\ifx\@revised\undefined\else\textcolor{rhocolor}{\bfseries\ignorespaces\revisedlanguage}\@revised\fi} 
                \hspace{10pt}
            {\ifx\@accepted\undefined\else\textcolor{rhocolor}{\bfseries\ignorespaces\acceptedlanguage}\@accepted\fi}
                \hspace{10pt}
            {\ifx\@published\undefined\else\textcolor{rhocolor}{\bfseries\ignorespaces\publishedlanguage}\@published\fi\par}
                \vskip3pt
            {\@license\par}
        \vskip6.5pt
    \hrule
}

%----------------------------------------------------------
% ABSTRACT STYLE
%----------------------------------------------------------

% Abstract text preferences commands
\newcommand{\keywords}[1]{\def\@keywords{#1}}
\newcommand{\absfont}{\normalfont\sffamily\linespread{1}\fontsize{8.5}{11}\selectfont}
\newcommand{\absheadfont}{\color{rhocolor}\normalfont\sffamily\fontsize{9}{11}\selectfont}
\newcommand{\keywordsfont}{\normalfont\sffamily\linespread{1}\fontsize{7.8}{9}\selectfont}
\newcommand{\keywordheadfont}{\normalfont\sffamily\fontsize{7.8}{9}\selectfont\bfseries}
\newcommand{\keywordname}{{\keywordheadfont Keywords:}}

\def\xabstract{abstract}
\long\def\abstract#1\end#2{
    \def\two{#2}\ifx\two\xabstract
    \long\gdef\theabstract{\ignorespaces#1}
    \def\go{\end{abstract}}
\else
    #1\end{#2}
    \gdef\theabstract{\vskip12pt 
    \vskip12pt}
    \let\go\relax\fi
\go}

\newcommand{\rhocontent}{
    \ignorespaces
    {
    {\absheadfont\bfseries\abstractname}\vskip0.5em\absfont\theabstract
    \vskip10pt
    {\keywordsfont\keywordname}\hspace*{0.1em} \keywordsfont \@keywords%
    \vskip10pt
    \rhoinfo
    }
}

%----------------------------------------------------------
% TITLE STYLE
%----------------------------------------------------------

\newcommand{\dates}[1]{\def\@dates{#1}}
\newcommand{\datesfont}{\fontsize{7.5}{9}\selectfont}

\renewcommand{\@maketitle}{
    \bgroup\setlength{\parindent}{0pt}
    {   
        \vskip-18pt
        {\RaggedRight\color{rhocolor}\sffamily\bfseries\fontsize{18}{22}\selectfont\@title\par}
        \vskip10pt
        {\RaggedRight\@author\par}
        \vskip8pt
        {\RaggedRight\ifx\@dates\undefined\else\datesfont\@dates\fi\par}
    \vskip20pt
        \rhocontent
    \vskip20pt
    }\egroup
}

%----------------------------------------------------------
% AFFILIATION SETUP
%----------------------------------------------------------

\usepackage{authblk}

\newcommand{\andlanguage}{
    \iflanguage{spanish}{
        { y }%
    }{%
        { and }%
    }%
}

\setlength{\affilsep}{9pt}
\renewcommand\Authfont{\fontsize{9}{11}\bfseries\sffamily\selectfont}
\renewcommand\Affilfont{\fontsize{7.5}{9}\it\selectfont}
\renewcommand\AB@affilsep{\reset@font\protect\Affilfont}
\renewcommand\AB@affilsepx{\reset@font\protect\\\protect\Affilfont}
% \renewcommand\AB@affilsepx{, \protect\Affilfont}

\renewcommand\Authand{\ignorespaces\andlanguage }
\renewcommand\Authands{\ignorespaces\andlanguage }

%----------------------------------------------------------
% SECTION STYLE
%----------------------------------------------------------

\setcounter{secnumdepth}{5}

\titleformat{\section}
  	{\color{rhocolor}\sffamily\large\bfseries}
  	{\thesection.}
  	{0.5em}
  	{#1}
  	[]

\titleformat{name=\section,numberless}[block]
        {\color{rhocolor}\sffamily\large\bfseries}
        {}
        {0em}
        {\tikz\draw[rhocolor, fill=rhocolor] (0,0) rectangle (6.6pt,6.6pt); \hspace{2.5pt} {#1}}
        []
  
\titleformat{\subsection}[block]
  	{\sffamily\bfseries}
  	{\thesubsection.}
  	{0.5em}
  	{#1}     
  	[]
  
\titleformat{\subsubsection}[block] % You may change it to "runin"
  	{\small\sffamily\bfseries\itshape}
  	{\thesubsubsection.}
  	{0.5em}
  	{#1}      % If using runin, change #1 to '#1. ' (space at the end)
  	[]    
  
\titleformat{\paragraph}[runin]
	{\small\bfseries}
  	{}
  	{0em}
  	{#1} 
  
\titlespacing*{\section}{0pc}{3ex \@plus4pt \@minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex \@plus3pt \@minus2pt}{2pt}
\titlespacing*{\subsubsection}{0pc}{2ex \@plus2.5pt \@minus1.5pt}{2pt}
\titlespacing*{\paragraph}{0pc}{1.5ex \@plus2pt \@minus1pt}{12pt}

%----------------------------------------------------------
% TABLE OF CONTENTS
%----------------------------------------------------------

\newlength{\tocsep} 
\setlength\tocsep{1.5pc} % Sets the indentation of the sections in the table of contents
\setcounter{tocdepth}{5} % Three levels in the table of contents section: sections, subsections and subsubsections

\usepackage{titletoc}
\contentsmargin{0cm}

\titlecontents{section}[\tocsep]
    {\addvspace{4pt}\sffamily\selectfont\bfseries}
    {\contentslabel[\thecontentslabel]{\tocsep}}
    {}
    {\hfill\thecontentspage}
    []

\titlecontents{subsection}[3pc]
    {\addvspace{4pt}\small\sffamily\selectfont}
    {\contentslabel[\thecontentslabel]{\tocsep}}
    {}
    {\ \titlerule*[.5pc]{.}\ \thecontentspage}
    []

\titlecontents*{subsubsection}[3pc]
    {\footnotesize\sffamily\selectfont}
    {}
    {}
    {}
    [\ \textbullet\ ]

%----------------------------------------------------------
% FIGURE-, TABLE-, LISTINGS- CAPTION STYLE
%----------------------------------------------------------

% Figure
\RequirePackage[
    labelfont={bf,sf},
    list=no,
    labelsep=period,
    singlelinecheck=off,
    font=small,
    justification=justified]
{caption}

% Table
\captionsetup[table]{position=below}
\newcommand{\addtabletext}[1]{{\setlength{\leftskip}{9pt}\fontsize{7}{9}\vskip2pt\selectfont#1}}

% Listings
\newcommand{\captioncodelanguage}{
    \iflanguage{spanish}{
        {C\'odigo}%
    }{%
        {Code}%
    }%
}
\captionsetup[lstlisting]{font=small, labelfont={bf,sf}, belowskip=3pt, position=below, labelsep=period}
\renewcommand\lstlistingname{\captioncodelanguage}
\renewcommand\lstlistlistingname{\captioncodelanguage}

%----------------------------------------------------------
% LISTINGS STYLE
%----------------------------------------------------------

% Defined colors for listings environment
\definecolor{rhocodeback}{RGB}{248, 248, 248}  
% \definecolor{rhocodeback}{RGB}{255, 255, 255}     % Alternative back color
\definecolor{rhocodecomment}{RGB}{1, 136, 0}
\definecolor{rhocodekey}{RGB}{53, 53, 128}
\definecolor{rhocodestring}{RGB}{122, 36, 47}
\definecolor{rhogray}{RGB}{0.5,0.5,0.5}
\definecolor{rhoredmatlab}{RGB}{199, 78, 0}

% Matlab
\lstset{
    language=Matlab,
    basicstyle=\small\ttfamily,
    backgroundcolor=\color{rhocodeback},   
    commentstyle=\color{rhocodecomment},
    keywordstyle=\color{rhocodekey},
    numberstyle=\tiny\color{rhogray},
    stringstyle=\color{rhocodestring},
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,          		
    numbers=left,                   % if "none" change the values
    numbersep=5pt,                  % 0pt
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    aboveskip=12pt,
    belowskip=8pt,
    xleftmargin=9pt,				% 0pt
    rulecolor=\color{rhogray},
    emph=[1]{for,end,break},
    emphstyle=[1]\color{rhoredmatlab},
    frame=lines
}

% General
\lstdefinestyle{rhocoding}{
    backgroundcolor=\color{rhocodeback},   
    commentstyle=\color{rhocodecomment},
    keywordstyle=\color{rhocodekey},
    numberstyle=\tiny\color{rhogray},
    stringstyle=\color{rhocodestring},
    breakatwhitespace=false,        
    basicstyle=\small\ttfamily,
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,				 	% if "none" change the values  
    numbersep=5pt,    				% 0pt              
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    aboveskip=12pt,
    belowskip=8pt,
    xleftmargin=9pt,				% 0pt
    rulecolor=\color{rhogray},
    frame=none
}

\lstset{style=rhocoding}

%----------------------------------------------------------
% FOOTNOTE STYLE
%----------------------------------------------------------

\definecolor{black50}{gray}{0.5}
\renewcommand*{\footnotelayout}{\normalfont\fontsize{6}{8}}
\renewcommand{\footnoterule}{
 	\kern -3pt
  	{\color{black50} \hrule width 72pt height 0.25pt}
  	\kern 2.5pt
}

%----------------------------------------------------------
% BIBLATEX
%----------------------------------------------------------

\usepackage[
    backend=biber,
    style=ieee,
    sorting=ynt
]{biblatex}

\addbibresource{rho.bib}

%----------------------------------------------------------

\endinput