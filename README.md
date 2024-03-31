# InfantCareAIChat

To start the project

1. Install the requirements.
2. Put documents in PDF format in `data` folder.
3. Get the Google Cloud credential JSON file following [this](https://stackoverflow.com/questions/46287267/how-can-i-get-the-file-service-account-json-for-google-translate-api) and put under `proof_of_concept`
4. Replace the credential file name in `os.environ['GOOGLE_APPLICATION_CREDENTIALS']` in `langchain_helper.py`.
5. Embed the documents by running `py document_parse_helper.py`.
6. Start the app by `streamlit run main.py`

...and have fun!
