mkdir -p code
mkdir -p data

# Download Data
echo "Downloading DocIDs"
wget -nc https://ir.nist.gov/covidSubmit/data/docids-rnd5.txt -P data
echo "Downloading topics"
wget -nc https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml -P data
echo "Downloading qrels"
wget -nc https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j4.5-5.txt -P data
echo "Downloading CORD-19"
wget -nc https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-07-16.tar.gz -P data

# Extract data
echo "Extracting CORD-19"
tar -zxvf data/cord-19_2020-07-16.tar.gz -C data
tar -zxvf data/2020-07-16/document_parses.tar.gz -C data/2020-07-16

# Run experiments
echo "Running experiments"
python code/experiments_cord.py
