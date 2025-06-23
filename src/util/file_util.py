import os                    # ← ADD THIS
import time                  # (already used for token expiry)
import mimetypes             # (used in _guess_mime_type)
import requests              # (used in _ensure_token, upload_file, etc.)
from pathlib import Path
from typing import Dict, Any, Optional, Union
# extra imports (put near the top of the file)
from http.client import HTTPSConnection
from urllib.parse import urlsplit
from io import BytesIO
from docx import Document          # pip install python-docx

# ──────────────────────────────────────────────────────────────────

class HermesFileUtil:
    """
    Convenience wrapper for Hermes file operations.

    🔧 Fill in `self.username`, `self.password`, and `self.account_id`,
    then just do: `hermes = HermesFileUtil()`.
    """

    # ---------- INITIALISATION -------------------------------------------------- #
    def __init__(self, *, env: str = "stage", timeout: int = 30) -> None:
        self.username: str = "fc_user@fuelcycle.com"
        self.password: str = "Test123!!"
        self.account_id: int = 1  # e.g. 1

        self.env = env
        self.timeout = timeout

        self._base_url = (
            f"https://hermes-{env}.fuelcycle{'' if env == 'prod' else env}.com"
            if env != "prod"
            else "https://hermes.fuelcycle.com"
        )
        self._access_token: Optional[str] = None
        self._token_expires: float = 0.0  # epoch seconds

        # extra MIME fall-backs for Office formats
        self._mime_fallbacks: Dict[str, str] = {
            ".csv": "text/csv",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".pdf": "application/pdf",
        }

    # ---------- PUBLIC API ------------------------------------------------------- #
    def upload_file(
        self,
        file_path: str | os.PathLike,
        *,
        reference_type: str = "GENERIC",
        mime_type: str | None = None,
    ) -> Dict[str, Any]:
        """Upload a local file and return Hermes’ JSON response."""
        token = self._ensure_token()

        file_path = Path(file_path)
        mime_type = mime_type or self._guess_mime_type(file_path)

        url = f"{self._base_url}/account/{self.account_id}/file"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"reference_type": reference_type}

        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, mime_type)}
            resp = requests.post(url, headers=headers, data=payload, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_file(
        self,
        file_id: Union[int, str],
        *,
        save_to: str | os.PathLike | None = None,
    ) -> Union[bytes, Dict[str, Any], Path]:
        """
        Hit Hermes’ `/file/{file_id}` endpoint.
        • If it returns JSON ⇒ metadata dict  
        • If it returns binary ⇒ bytes (or save to disk if `save_to` is given)
        """
        token = self._ensure_token()

        url = f"{self._base_url}/account/{self.account_id}/file/{file_id}"
        headers = {"Authorization": f"Bearer {token}"}

        resp = requests.get(url, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        if resp.headers.get("Content-Type", "").startswith("application/json"):
            return resp.json()

        content = resp.content
        if save_to is None:
            return content

        save_path = Path(save_to)
        if save_path.is_dir():
            fname = self._extract_filename(resp.headers.get("Content-Disposition")) or f"file_{file_id}"
            save_path = save_path / fname
        save_path.write_bytes(content)
        return save_path

    def fetch_downloadable_url(self, url: str) -> bytes:
        """
        Download bytes from a presigned S3 URL *without* path normalisation.
        Works even when the link contains a double slash (`//`) that would
        otherwise break the signature.
        """
        parsed = urlsplit(url)
        if parsed.scheme != "https":
            raise ValueError("Only https:// URLs are supported")
    
        # Build exact path + query that S3 signed
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
    
        conn = HTTPSConnection(parsed.netloc, timeout=self.timeout)
        conn.request("GET", path, headers={"Host": parsed.netloc})
        resp = conn.getresponse()
    
        if resp.status != 200:                       # S3 returns XML on error
            snippet = resp.read(300).decode(errors="ignore")
            conn.close()
            raise RuntimeError(f"S3 responded {resp.status}: {snippet}")
    
        data = resp.read()
        conn.close()
        return data
    
    def read_docx_text(self, file_id: int | str) -> str:
        """
        Return the plain text of a .docx stored in Hermes (identified by
        `file_id`).  Raises if the file isn’t a DOCX or can’t be downloaded.
        """
        meta = self.get_file(file_id)
        if not isinstance(meta, dict):
            raise ValueError("Expected metadata JSON but got raw bytes")
    
        if meta["data"]["file_type"].lower() != ".docx":
            raise ValueError("File is not a .docx document")
    
        url = meta["data"]["downloadable_url"]
        content = self.fetch_downloadable_url(url)
    
        doc = Document(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    def download_file_content(self, file_id: Union[int, str]) -> bytes:
        """
        Convenience wrapper: look up metadata, pull `downloadable_url`,
        then fetch that URL’s bytes (in-memory).
        """
        meta = self.get_file(file_id)
        if not isinstance(meta, dict):
            raise ValueError("Expected metadata JSON but got raw bytes—did you pass the wrong ID?")

        url = meta.get("data", {}).get("downloadable_url")
        if not url:
            raise ValueError("Metadata JSON did not include 'downloadable_url'")

        return self.fetch_downloadable_url(url)

    # ---------- INTERNAL HELPERS ------------------------------------------------- #
    def _ensure_token(self) -> str:
        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        auth_url = f"{self._base_url}/auth/login"
        resp = requests.post(
            auth_url,
            json={"username": self.username, "password": self.password},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]["auth_result"]
        self._access_token = data["AccessToken"]
        self._token_expires = time.time() + int(data.get("ExpiresIn", 0))
        return self._access_token

    def _guess_mime_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(path.name)
        return mime or self._mime_fallbacks.get(path.suffix.lower(), "application/octet-stream")

    @staticmethod
    def _extract_filename(content_disposition: Optional[str]) -> Optional[str]:
        if not content_disposition:
            return None
        for part in content_disposition.split(";"):
            if part.strip().startswith("filename="):
                return part.split("=", 1)[1].strip().strip('"')
        return None
