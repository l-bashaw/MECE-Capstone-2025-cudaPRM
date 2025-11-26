.PHONY: root.pdf all clean
all: root.pdf root_opt.pdf

root.pdf: root.tex Makefile
	latexmk -pdf -f -bibtex -pdflatex="pdflatex -interaction=nonstopmode" -shell-escape -use-make root.tex 

# Optimize the paper for submission
root_opt.pdf: root.pdf
	ghostscript -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=root_opt.pdf root.pdf
clean:
	latexmk -CA
	rm root.*[^tex]

