# latexmk config per study/docs
$pdf_mode = 1;            # pdflatex
$out_dir  = 'build';
$aux_dir  = 'build';
@default_files = ('main.tex');
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
