mkdir -p code
mkdir -p data

# Download data
echo "Downloading pmc-00.tar.gz"
wget -nc http://ceb.nlm.nih.gov/~robertske/pmc-00.tar.gz -P data
echo "Downloading pmc-01.tar.gz"
wget -nc http://ceb.nlm.nih.gov/~robertske/pmc-01.tar.gz -P data
echo "Downloading pmc-02.tar.gz"
wget -nc http://ceb.nlm.nih.gov/~robertske/pmc-02.tar.gz -P data
echo "Downloading pmc-03.tar.gz"
wget -nc http://ceb.nlm.nih.gov/~robertske/pmc-03.tar.gz -P data

# Download topics
echo "Downloading topics2016.xml"
wget -nc http://www.trec-cds.org/topics2016.xml -P data

# Download evaluation materials
echo "Downloading qrels-treceval-2016.txt"
wget -nc https://trec.nist.gov/data/clinical/qrels-treceval-2016.txt -P data
echo "Downloading qrels-sampleval-2016.txt"
wget -nc https://trec.nist.gov/data/clinical/qrels-sampleval-2016.txt -P data
echo "Downloading sample_eval.pl"
wget -nc https://trec.nist.gov/data/clinical/sample_eval.pl -P code
echo "Downloading trec_eval-9.0.7.tar.gz"
wget -nc https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz -P code

# Extract data
echo "Extracting pmc-00.tar.gz"
tar -zxvf data/pmc-00.tar.gz -C data
echo "Extracting pmc-01.tar.gz"
tar -zxvf data/pmc-01.tar.gz -C data
echo "Extracting pmc-02.tar.gz"
tar -zxvf data/pmc-02.tar.gz -C Data
echo "Extracting pmc-03.tar.gz"
tar -zxvf data/pmc-03.tar.gz -C data

# Consolidate data
mkdir -p data/pmc2016
for subdir in data/pmc-*/*; do
  mv ${subdir}/*.nxml data/pmc2016/.
done

# Run experiments
echo "Running experiments"
python code/experiments_2016.py

# Run evaluation
#code/trec_eval.pl ....
