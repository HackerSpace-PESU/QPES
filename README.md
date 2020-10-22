# QPES
The QA Bot for PESU built by HackerSpace!

## Setting up Environment
1. Clone the repo
```bash
git clone https://github.com/HackerSpace-PESU/QPES
```
2. Run the script to setup the virtual environments (with package installations)
```bash
cd QPES/
chmod +x setup-environment.sh
sudo ./setup-environment.sh
```
## Testing QPES
Since QPES is a closed domain QnA System, all information needs to be present in the `pes-corpus.txt` document in the `data/` directory. Additionally, all questions need to be added to `questions.txt` in the `test-questions/` directory. Once these are setup, run
```bash
sudo ./test-models.sh
```