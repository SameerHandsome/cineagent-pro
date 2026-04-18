import ssl

import redis

r = redis.from_url(
    'rediss://:AWDnAAIncDIyODQ4YmVhOGM2M2Y0NWE3ODU0M2FlOGIxMzU0YWY2ZXAyMjQ4MDc@exact-caribou-24807.upstash.io:6380/0',
    ssl_cert_reqs=ssl.CERT_NONE
)
print(r.ping())
