# This report needs XeLaTeX (fontspec + Arial). Editors such as VS Code's
# LaTeX Workshop call `latexmk -pdf`, which would run pdflatex; route that
# to xelatex so the default build recipe works too.
$pdf_mode = 5;
$pdflatex = 'xelatex %O %S';
