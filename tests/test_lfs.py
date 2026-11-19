import os
import threading
import time
from hashlib import md5, sha256
from io import BytesIO
from math import ceil
from unittest.mock import MagicMock

import pytest

from huggingface_hub import constants
from huggingface_hub._commit_api import CommitOperationAdd
from huggingface_hub.lfs import UploadInfo, _upload_multi_part, post_lfs_batch_info
from huggingface_hub.utils import SoftTemporaryDirectory
from huggingface_hub.utils._lfs import SliceFileObj


class TestUploadInfo:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.content = b"RandOm ConTEnT" * 1024
        self.size = len(self.content)
        self.sha = sha256(self.content).digest()
        self.sample = self.content[:512]

    def test_upload_info_from_path(self):
        with SoftTemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "file.bin")
            with open(filepath, "wb+") as file:
                file.write(self.content)
            upload_info = UploadInfo.from_path(filepath)

            assert upload_info.sample == self.sample
            assert upload_info.size == self.size
            # sha256 is lazy: computed on first access, requires the source file to still exist
            assert not upload_info.is_hashed
            assert upload_info.sha256 == self.sha
            assert upload_info.is_hashed

    def test_upload_info_from_bytes(self):
        upload_info = UploadInfo.from_bytes(self.content)

        assert upload_info.sample == self.sample
        assert upload_info.size == self.size
        assert upload_info.sha256 == self.sha

    def test_upload_info_from_bytes_io(self):
        upload_info = UploadInfo.from_fileobj(BytesIO(self.content))

        assert upload_info.sample == self.sample
        assert upload_info.size == self.size
        assert upload_info.sha256 == self.sha


class TestSliceFileObj:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.content = b"RANDOM self.content uauabciabeubahveb" * 1024

    def test_slice_fileobj_BytesIO(self):
        fileobj = BytesIO(self.content)
        prev_pos = fileobj.tell()

        # Test read
        with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
            assert fileobj_slice.tell() == 0
            assert fileobj_slice.read() == self.content[24:42]
            assert fileobj_slice.tell() == 18
            assert fileobj_slice.read() == b""
            assert fileobj_slice.tell() == 18

        assert fileobj.tell() == prev_pos

        with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
            assert fileobj_slice.tell() == 0
            assert fileobj_slice.read(200) == self.content[0:200]
            assert fileobj_slice.read(500) == self.content[200:700]
            assert fileobj_slice.read(200) == self.content[700:900]
            assert fileobj_slice.read(200) == self.content[900:990]
            assert fileobj_slice.read(200) == b""

        # Test seek with whence = os.SEEK_SET
        with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
            assert fileobj_slice.tell() == 0
            fileobj_slice.seek(2, os.SEEK_SET)
            assert fileobj_slice.tell() == 2
            assert fileobj_slice.fileobj.tell() == 102
            fileobj_slice.seek(-4, os.SEEK_SET)
            assert fileobj_slice.tell() == 0
            assert fileobj_slice.fileobj.tell() == 100
            fileobj_slice.seek(100 + 4, os.SEEK_SET)
            assert fileobj_slice.tell() == 100
            assert fileobj_slice.fileobj.tell() == 200

        # Test seek with whence = os.SEEK_CUR
        with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
            assert fileobj_slice.tell() == 0
            fileobj_slice.seek(-5, os.SEEK_CUR)
            assert fileobj_slice.tell() == 0
            assert fileobj_slice.fileobj.tell() == 100
            fileobj_slice.seek(50, os.SEEK_CUR)
            assert fileobj_slice.tell() == 50
            assert fileobj_slice.fileobj.tell() == 150
            fileobj_slice.seek(100, os.SEEK_CUR)
            assert fileobj_slice.tell() == 100
            assert fileobj_slice.fileobj.tell() == 200
            fileobj_slice.seek(-300, os.SEEK_CUR)
            assert fileobj_slice.tell() == 0
            assert fileobj_slice.fileobj.tell() == 100

        # Test seek with whence = os.SEEK_END
        with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
            assert fileobj_slice.tell() == 0
            fileobj_slice.seek(-5, os.SEEK_END)
            assert fileobj_slice.tell() == 95
            assert fileobj_slice.fileobj.tell() == 195
            fileobj_slice.seek(50, os.SEEK_END)
            assert fileobj_slice.tell() == 100
            assert fileobj_slice.fileobj.tell() == 200
            fileobj_slice.seek(-200, os.SEEK_END)
            assert fileobj_slice.tell() == 0
            assert fileobj_slice.fileobj.tell() == 100

    def test_slice_fileobj_iter_reads_full_slice(self):
        # Regression test: iterating a slice must yield the whole slice in 4 MB
        # chunks. Previously __iter__ had a single `yield`, so it returned only the
        # first 4 MB chunk and silently dropped everything after it.
        chunk_size = 4 * 1024 * 1024
        content = b"x" * (chunk_size + 500)  # spans more than one chunk
        fileobj = BytesIO(content)
        with SliceFileObj(fileobj, seek_from=0, read_limit=len(content)) as fileobj_slice:
            chunks = list(fileobj_slice)
        assert len(chunks) > 1
        assert b"".join(chunks) == content

    def test_slice_fileobj_file(self):
        self.content = b"RANDOM self.content uauabciabeubahveb" * 1024

        with SoftTemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "file.bin")
            with open(filepath, "wb+") as f:
                f.write(self.content)
            with open(filepath, "rb") as fileobj:
                prev_pos = fileobj.tell()
                # Test read
                with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
                    assert fileobj_slice.tell() == 0
                    assert fileobj_slice.read() == self.content[24:42]
                    assert fileobj_slice.tell() == 18
                    assert fileobj_slice.read() == b""
                    assert fileobj_slice.tell() == 18

                assert fileobj.tell() == prev_pos

                with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
                    assert fileobj_slice.tell() == 0
                    assert fileobj_slice.read(200) == self.content[0:200]
                    assert fileobj_slice.read(500) == self.content[200:700]
                    assert fileobj_slice.read(200) == self.content[700:900]
                    assert fileobj_slice.read(200) == self.content[900:990]
                    assert fileobj_slice.read(200) == b""

                # Test seek with whence = os.SEEK_SET
                with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                    assert fileobj_slice.tell() == 0
                    fileobj_slice.seek(2, os.SEEK_SET)
                    assert fileobj_slice.tell() == 2
                    assert fileobj_slice.fileobj.tell() == 102
                    fileobj_slice.seek(-4, os.SEEK_SET)
                    assert fileobj_slice.tell() == 0
                    assert fileobj_slice.fileobj.tell() == 100
                    fileobj_slice.seek(100 + 4, os.SEEK_SET)
                    assert fileobj_slice.tell() == 100
                    assert fileobj_slice.fileobj.tell() == 200

                # Test seek with whence = os.SEEK_CUR
                with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                    assert fileobj_slice.tell() == 0
                    fileobj_slice.seek(-5, os.SEEK_CUR)
                    assert fileobj_slice.tell() == 0
                    assert fileobj_slice.fileobj.tell() == 100
                    fileobj_slice.seek(50, os.SEEK_CUR)
                    assert fileobj_slice.tell() == 50
                    assert fileobj_slice.fileobj.tell() == 150
                    fileobj_slice.seek(100, os.SEEK_CUR)
                    assert fileobj_slice.tell() == 100
                    assert fileobj_slice.fileobj.tell() == 200
                    fileobj_slice.seek(-300, os.SEEK_CUR)
                    assert fileobj_slice.tell() == 0
                    assert fileobj_slice.fileobj.tell() == 100

                # Test seek with whence = os.SEEK_END
                with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                    assert fileobj_slice.tell() == 0
                    fileobj_slice.seek(-5, os.SEEK_END)
                    assert fileobj_slice.tell() == 95
                    assert fileobj_slice.fileobj.tell() == 195
                    fileobj_slice.seek(50, os.SEEK_END)
                    assert fileobj_slice.tell() == 100
                    assert fileobj_slice.fileobj.tell() == 200
                    fileobj_slice.seek(-200, os.SEEK_END)
                    assert fileobj_slice.tell() == 0
                    assert fileobj_slice.fileobj.tell() == 100


def test_post_lfs_batch_info_uses_http_backoff(mocker):
    """post_lfs_batch_info uses http_backoff for retry on transient failures."""
    mock_http_backoff = mocker.patch("huggingface_hub.lfs.http_backoff")
    mocker.patch("huggingface_hub.lfs.hf_raise_for_status")
    mock_http_backoff.return_value = MagicMock(json=lambda: {"objects": []})

    post_lfs_batch_info(
        upload_infos=[UploadInfo(sha256=b"\x00" * 32, size=100, sample=b"test")],
        token="test_token",
        repo_type="model",
        repo_id="test/repo",
    )

    mock_http_backoff.assert_called_once()
    assert mock_http_backoff.call_args[0][0] == "POST"
    assert "/info/lfs/objects/batch" in mock_http_backoff.call_args[0][1]


class TestMultipartUpload:
    """Multipart LFS upload: concurrent parts, MD5-verified, resumable after a network drop."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        # Persist resume state in a tmp cache dir
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "cache"))

        self.chunk_size = 1024
        self.content = b"R4nd0m CoNtEnT!" * 1024  # 15 parts of 1KB
        self.filepath = tmp_path / "file.bin"
        self.filepath.write_bytes(self.content)
        self.operation = CommitOperationAdd(path_in_repo="file.bin", path_or_fileobj=str(self.filepath))
        self.num_parts = ceil(len(self.content) / self.chunk_size)
        self.part_urls = [f"https://example.com/upload/part{idx}" for idx in range(self.num_parts)]
        self.header = {"chunk_size": str(self.chunk_size)}
        self.header.update({str(idx + 1): url for idx, url in enumerate(self.part_urls)})
        self.upload_url = "https://example.com/upload/complete"

        # Record HTTP calls made by the upload
        self.put_parts: list[int] = []
        self.put_thread_ids: set[int] = set()
        self.completion_payloads: list[dict] = []
        self.fail_from_part: int | None = None  # simulate a network drop from this part onwards

        def fake_http_backoff(method, url, **kwargs):
            if method == "PUT":
                part_idx = self.part_urls.index(url)
                if self.fail_from_part is not None and part_idx >= self.fail_from_part:
                    raise ConnectionError("Simulated network drop")
                data = kwargs["data"].read()
                assert data == self.content[part_idx * self.chunk_size : (part_idx + 1) * self.chunk_size]
                time.sleep(0.01)  # ensure parts actually overlap when uploaded concurrently
                self.put_parts.append(part_idx)
                self.put_thread_ids.add(threading.get_ident())
                return MagicMock(headers={"etag": md5(data).hexdigest()})
            assert method == "POST"
            self.completion_payloads.append(kwargs["json"])
            return MagicMock()

        monkeypatch.setattr("huggingface_hub.lfs.http_backoff", fake_http_backoff)
        monkeypatch.setattr("huggingface_hub.lfs.hf_raise_for_status", lambda response: None)

    def _state_path(self) -> str:
        oid = self.operation.upload_info.sha256.hex()
        return os.path.join(constants.HF_HUB_CACHE, "uploads", f"{oid}.json")

    def test_resume_after_connection_drop(self):
        # 1. First attempt: network drops after 6 parts (out of 15)
        self.fail_from_part = 6
        with pytest.raises(ConnectionError, match="Simulated network drop"):
            _upload_multi_part(self.operation, self.header, self.chunk_size, self.upload_url)

        uploaded_parts = sorted(set(self.put_parts))
        assert uploaded_parts == list(range(6))
        assert os.path.exists(self._state_path())  # resume state persisted

        # 2. Second attempt: network is back, upload resumes where it stopped
        self.put_parts.clear()
        self.fail_from_part = None
        _upload_multi_part(self.operation, self.header, self.chunk_size, self.upload_url)

        # Only the missing parts are re-uploaded
        assert sorted(set(self.put_parts)) == list(range(6, self.num_parts))
        # Parts were uploaded concurrently
        assert len(self.put_thread_ids) > 1
        # Completion request sent once, with all 15 part etags in order
        assert len(self.completion_payloads) == 1
        payload = self.completion_payloads[0]
        assert payload["oid"] == self.operation.upload_info.sha256.hex()
        assert [part["partNumber"] for part in payload["parts"]] == list(range(1, self.num_parts + 1))
        assert [part["etag"] for part in payload["parts"]] == [
            md5(self.content[idx * self.chunk_size : (idx + 1) * self.chunk_size]).hexdigest()
            for idx in range(self.num_parts)
        ]
        # Resume state cleaned up after successful upload
        assert not os.path.exists(self._state_path())

    def test_part_hash_mismatch_raises(self, monkeypatch):
        # Server acknowledges the upload but returns an etag that doesn't match the part content
        def corrupted_http_backoff(method, url, **kwargs):
            if method == "PUT":
                return MagicMock(headers={"etag": "0" * 32})
            return MagicMock()

        monkeypatch.setattr("huggingface_hub.lfs.http_backoff", corrupted_http_backoff)

        with pytest.raises(ValueError, match="got corrupted during upload"):
            _upload_multi_part(self.operation, self.header, self.chunk_size, self.upload_url)
