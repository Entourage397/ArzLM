"""Hugging Face Hub metadata client. Never downloads file bytes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from arzlm.data.security.policy import RemoteFileMeta


class HubClient(Protocol):
    def resolve_revision(self, dataset_id: str, revision: str) -> str: ...

    def get_file(self, dataset_id: str, revision: str, path: str) -> RemoteFileMeta: ...

    def list_files(
        self,
        dataset_id: str,
        revision: str,
        prefix: str,
        *,
        suffixes: tuple[str, ...] | None = None,
    ) -> list[RemoteFileMeta]: ...


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "items"):
        try:
            return dict(obj)
        except Exception:
            pass
    out: dict[str, Any] = {}
    for key in ("status", "message", "version", "pickleImports", "reportLink"):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def meta_from_repo_file(dataset_id: str, revision: str, item: Any) -> RemoteFileMeta | None:
    path = getattr(item, "path", None) or getattr(item, "rfilename", None)
    if not path:
        return None
    path = str(path).replace("\\", "/")
    if path.endswith("/"):
        return None
    size = getattr(item, "size", None)
    lfs = getattr(item, "lfs", None)
    sha256 = None
    if lfs is not None:
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    xet = getattr(item, "xet_hash", None) or getattr(item, "xetHash", None)
    blob_id = getattr(item, "blob_id", None) or getattr(item, "oid", None)
    sec = getattr(item, "security", None)
    status = av_status = av_message = pickle_status = None
    extra: dict[str, Any] = {}
    if sec is not None:
        extra["security"] = _as_dict(sec)
        status = getattr(sec, "status", None)
        if status is None and isinstance(sec, dict):
            status = sec.get("status")
        av = getattr(sec, "av_scan", None)
        if av is None and isinstance(sec, dict):
            av = sec.get("av_scan") or sec.get("avScan")
        avd = _as_dict(av)
        av_status = avd.get("status")
        av_message = avd.get("message")
        pk = getattr(sec, "pickle_import_scan", None)
        if pk is None and isinstance(sec, dict):
            pk = sec.get("pickle_import_scan") or sec.get("pickleImportScan")
        pickle_status = _as_dict(pk).get("status")
    return RemoteFileMeta(
        dataset_id=dataset_id,
        revision=revision,
        path=path,
        size=int(size) if size is not None else None,
        sha256=str(sha256) if sha256 else None,
        xet_hash=str(xet) if xet else None,
        blob_id=str(blob_id) if blob_id else None,
        security_status=str(status) if status else None,
        av_status=str(av_status) if av_status else None,
        av_message=str(av_message) if av_message else None,
        pickle_status=str(pickle_status) if pickle_status else None,
        extra=extra,
    )


class HfHubClient:
    """Live Hub metadata. Uses list_repo_tree / get_paths_info / dataset_info only."""

    def __init__(self, api: Any | None = None) -> None:
        self._api = api

    def _api_obj(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    def resolve_revision(self, dataset_id: str, revision: str) -> str:
        info = self._api_obj().dataset_info(dataset_id, revision=revision)
        sha = getattr(info, "sha", None)
        if not sha:
            raise RuntimeError(f"dataset_info({dataset_id!r} @ {revision}) returned no sha")
        return str(sha)

    def get_file(self, dataset_id: str, revision: str, path: str) -> RemoteFileMeta:
        items = self._api_obj().get_paths_info(
            dataset_id,
            [path],
            repo_type="dataset",
            revision=revision,
            expand=True,
        )
        if not items:
            raise FileNotFoundError(f"{dataset_id} {path} @ {revision}")
        meta = meta_from_repo_file(dataset_id, revision, items[0])
        if meta is None:
            raise FileNotFoundError(f"{dataset_id} {path} @ {revision} is not a file")
        return meta

    def list_files(
        self,
        dataset_id: str,
        revision: str,
        prefix: str,
        *,
        suffixes: tuple[str, ...] | None = None,
    ) -> list[RemoteFileMeta]:
        prefix = prefix.replace("\\", "/").lstrip("/")
        try:
            out = self._list_files_tree(dataset_id, revision, prefix, suffixes=suffixes)
            if out:
                return out
        except Exception as exc:
            print(f"list_repo_tree failed for {dataset_id} {prefix!r}: {exc!r}", flush=True)
        out = self._list_files_by_name(dataset_id, revision, prefix, suffixes=suffixes)
        print(
            f"listed {dataset_id} {prefix!r} via list_repo_files: {len(out)} files",
            flush=True,
        )
        return out

    def _list_files_tree(
        self,
        dataset_id: str,
        revision: str,
        prefix: str,
        *,
        suffixes: tuple[str, ...] | None,
    ) -> list[RemoteFileMeta]:
        path_in_repo = prefix.rstrip("/") if prefix else None
        out: list[RemoteFileMeta] = []
        tree: Iterable[Any] = self._api_obj().list_repo_tree(
            dataset_id,
            path_in_repo=path_in_repo,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
        for item in tree:
            meta = meta_from_repo_file(dataset_id, revision, item)
            if meta is None:
                continue
            rel = meta.path
            if prefix and not (
                rel == prefix.rstrip("/")
                or rel.startswith(prefix if prefix.endswith("/") else prefix + "/")
            ):
                continue
            if suffixes and not any(rel.endswith(s) for s in suffixes):
                continue
            out.append(meta)
        out.sort(key=lambda m: m.path)
        return out

    def _list_files_by_name(
        self,
        dataset_id: str,
        revision: str,
        prefix: str,
        *,
        suffixes: tuple[str, ...] | None,
        batch: int = 40,
    ) -> list[RemoteFileMeta]:
        names = list(self._api_obj().list_repo_files(dataset_id, revision=revision, repo_type="dataset"))
        wanted: list[str] = []
        for rel in names:
            rel = rel.replace("\\", "/")
            if prefix and not (
                rel == prefix.rstrip("/")
                or rel.startswith(prefix if prefix.endswith("/") else prefix + "/")
            ):
                continue
            if suffixes and not any(rel.endswith(s) for s in suffixes):
                continue
            wanted.append(rel)
        out: list[RemoteFileMeta] = []
        for i in range(0, len(wanted), batch):
            chunk = wanted[i : i + batch]
            items = self._api_obj().get_paths_info(
                dataset_id,
                chunk,
                repo_type="dataset",
                revision=revision,
                expand=True,
            )
            for item in items:
                meta = meta_from_repo_file(dataset_id, revision, item)
                if meta is not None:
                    out.append(meta)
        out.sort(key=lambda m: m.path)
        return out


class FakeHubClient:
    """In-memory Hub client for tests. Never touches the network."""

    def __init__(
        self,
        files: Sequence[RemoteFileMeta],
        *,
        resolved: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.files = list(files)
        self.resolved = dict(resolved or {})
        self.calls: list[tuple[str, ...]] = []

    def resolve_revision(self, dataset_id: str, revision: str) -> str:
        self.calls.append(("resolve", dataset_id, revision))
        return self.resolved.get((dataset_id, revision), revision)

    def get_file(self, dataset_id: str, revision: str, path: str) -> RemoteFileMeta:
        self.calls.append(("get", dataset_id, revision, path))
        path = path.replace("\\", "/")
        for meta in self.files:
            if meta.dataset_id == dataset_id and meta.path.replace("\\", "/") == path:
                if revision in {meta.revision, "main"} or not revision:
                    return meta
                if meta.revision == revision:
                    return meta
                return meta
        raise FileNotFoundError(f"{dataset_id} {path}")

    def list_files(
        self,
        dataset_id: str,
        revision: str,
        prefix: str,
        *,
        suffixes: tuple[str, ...] | None = None,
    ) -> list[RemoteFileMeta]:
        self.calls.append(("list", dataset_id, revision, prefix))
        prefix = (prefix or "").replace("\\", "/")
        resolved = self.resolved.get((dataset_id, revision), revision)
        out: list[RemoteFileMeta] = []
        for meta in self.files:
            if meta.dataset_id != dataset_id:
                continue
            if revision and meta.revision not in {revision, resolved}:
                continue
            rel = meta.path.replace("\\", "/")
            if prefix:
                needle = prefix if prefix.endswith("/") else prefix + "/"
                if not (rel == prefix.rstrip("/") or rel.startswith(needle)):
                    continue
            if suffixes and not any(rel.endswith(s) for s in suffixes):
                continue
            out.append(meta)
        out.sort(key=lambda m: m.path)
        return out
