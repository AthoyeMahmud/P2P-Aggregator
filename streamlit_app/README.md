# qBit Plugins Streamlit Aggregator

This app loads the qBittorrent search plugins from `../plugins` and aggregates
results in a Streamlit UI.

Run (from repo root):

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Notes:
- Some engines require accounts or local configuration files (errors show in UI).
- Network access is required to query trackers and indexers.
