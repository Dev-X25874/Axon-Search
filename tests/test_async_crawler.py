from __future__ import annotations

from crawler.async_crawler import AsyncCrawler, _is_private_host


def test_public_hostname_is_not_private():
    # This resolves publicly in real DNS; guard against flaky CI by also
    # allowing "unresolvable" to pass through (see _is_private_host docstring).
    assert _is_private_host("example.com") is False


def test_loopback_ip_is_private():
    assert _is_private_host("127.0.0.1") is True


def test_link_local_metadata_ip_is_private():
    # Cloud metadata endpoint commonly targeted by SSRF payloads.
    assert _is_private_host("169.254.169.254") is True


def test_rfc1918_ip_is_private():
    assert _is_private_host("10.0.0.5") is True
    assert _is_private_host("192.168.1.1") is True
    assert _is_private_host("172.16.0.1") is True


def test_localhost_hostname_is_private():
    assert _is_private_host("localhost") is True


def test_is_allowed_url_rejects_non_http_scheme():
    crawler = AsyncCrawler()
    assert crawler._is_allowed_url("ftp://example.com/file") is False
    assert crawler._is_allowed_url("javascript:alert(1)") is False


def test_is_allowed_url_rejects_disallowed_extension():
    crawler = AsyncCrawler()
    assert crawler._is_allowed_url("http://example.com/image.png") is False
    assert crawler._is_allowed_url("http://example.com/doc.pdf") is False


def test_is_allowed_url_rejects_private_target():
    crawler = AsyncCrawler()
    assert crawler._is_allowed_url("http://127.0.0.1/admin") is False
    assert crawler._is_allowed_url("http://169.254.169.254/latest/meta-data/") is False


def test_is_allowed_url_respects_allowed_domains():
    crawler = AsyncCrawler(allowed_domains=["arxiv.org"])
    assert crawler._is_allowed_url("http://arxiv.org/abs/1234") is True
    assert crawler._is_allowed_url("http://other.test/page") is False


def test_is_allowed_url_accepts_ordinary_public_url():
    crawler = AsyncCrawler()
    assert crawler._is_allowed_url("http://example.com/some/page") is True
