# latexmk config per study/docs
$pdf_mode = 1;            # pdflatex
$out_dir  = 'build';
$aux_dir  = 'build';
@default_files = ('exercises.tex', 'models_and_pricing.tex', 'architecture.tex');
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
