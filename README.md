# SEE
This repository introduces the self-evaluated-expertise (SEE) measure in experimental physics, a measure of student's physics self-recognition. 

## Preliminaries
The repository extracts the latest ECLASS data and from this constructs the SEE score. In order for the SEE repo to access the latest ECLASS data, it is required to clone the `eclass-public` repo, which can be done using the following command in the terminal:

```bash
git clone https://github.com/Lewandowski-Labs-PER/eclass-public.git
```

## Outline
The repo produces the following routines and figures as seen in the paper:

1) Donwload and plots the latest ECLASS data (`ECLASS-Violin-Plots.r`)
2) Filter the SEE data (`SEE-Filtering.ipynb`)
3) Compare and analyse the connection of ECLASS to SEE.
