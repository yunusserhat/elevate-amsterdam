# -*- coding: utf-8 -*-
"""Step 8.6: publish the search API Space (run only with the user's approval).

Creates the public Docker Space yunusserhat/elevate-amsterdam (if it does not exist) and uploads
dist/space_api (built by 85_package.py). Uses the token of the logged-in user; prints no secrets.
Usage: python 86_publish_space.py
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "yunusserhat/elevate-amsterdam"
SRC = Path(__file__).resolve().parents[1] / "dist/space_api"

api = HfApi()
print("user:", api.whoami()["name"])
url = api.create_repo(REPO, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
print("space:", url)
info = api.upload_folder(folder_path=str(SRC), repo_id=REPO, repo_type="space",
                         commit_message="Search API and pilot survey endpoints for ELEVATE in Amsterdam")
print("commit:", info.oid if hasattr(info, "oid") else info)
print("files:", sorted(api.list_repo_files(REPO, repo_type="space")))
