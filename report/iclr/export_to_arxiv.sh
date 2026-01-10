# export_to_arxiv.sh
#!/bin/bash
tectonic iclr2025.tex --keep-logs --keep-intermediates

zip icml_arxiv.zip iclr2025.tex iclr2025.bib icml2025.sty icml2025.bst algorithmic.sty algorithm.sty fancyhdr.sty
