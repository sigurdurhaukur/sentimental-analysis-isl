mkdir data

wget https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1 -O MTL_grouped.zip
unzip MTL_grouped.zip
find . -maxdepth 1 -type f -name "*.tsv" ! -name "is.tsv" -exec rm {} +

mv is.tsv data/is.tsv  


curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/124{/rmh_filters.zip}
unzip -q rmh_filters.zip

huggingface-cli login
