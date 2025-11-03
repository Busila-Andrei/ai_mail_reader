from __future__ import annotations

"""
Check Microsoft Graph via device-code flow and list top 5 message subjects.
Use environment variable GRAPH_CLIENT_ID with an Azure App (Public client) that has Mail.Read (Delegated) permission.
No password needed; you'll receive a code to enter in a browser.
"""

import os
import msal
import requests


def main() -> None:
    client_id = os.getenv("GRAPH_CLIENT_ID")
    if not client_id:
        print("Set GRAPH_CLIENT_ID env var to your Azure AD App's Client ID.")
        return
    app = msal.PublicClientApplication(client_id, authority="https://login.microsoftonline.com/common")
    flow = app.initiate_device_flow(scopes=["Mail.Read"])
    if "user_code" not in flow:
        print("Device flow failed to start.")
        return
    print(flow["message"])  # Instructions to authenticate
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        print("Failed to acquire token:", result)
        return
    r = requests.get(
        "https://graph.microsoft.com/v1.0/me/messages?$top=5",
        headers={"Authorization": f"Bearer {result['access_token']}"}
    )
    if r.status_code != 200:
        print("Graph error:", r.status_code, r.text)
        return
    for m in r.json().get("value", []):
        print(m.get("subject", ""))


if __name__ == "__main__":
    main()


