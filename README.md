# InfantCareAIChat

To start the project

Under `proof_of_concept` folder

1. Set up your Python 3.12 environments with venv Install the requirements by `pip install -r requirements.txt`.
2. Put documents in PDF format in `data` folder.
3. Get the Google Cloud credential JSON file following [this](https://stackoverflow.com/questions/46287267/how-can-i-get-the-file-service-account-json-for-google-translate-api) and put under `proof_of_concept`
4. Replace the value of `os.environ['GOOGLE_APPLICATION_CREDENTIALS']` in `langchain_helper.py` with the downloaded file name.
5. Embed the documents by running `python document_parse_helper.py`. Your embedding files should show up under the `embeddings` folder.
5.1 You can create another smaller embeddings with different settings, for backup using similar way, and rename it to `faiss_embedding_small.pk1` and place to `embeddings` folder as well.
6. Start the app by running `streamlit run main.py`

...and have fun!

![image](https://github.com/isabellachen3988/InfantCareAIChat/assets/50829041/d3e6644a-5f90-4a1d-be1c-7068092d063a)
