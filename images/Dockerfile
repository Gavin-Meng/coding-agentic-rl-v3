FROM docker.io/library/node@sha256:16d364eebf6b62da439dc993d9b80940c78b0ca38438452f011ab9a25c752644 AS node
FROM docker.io/library/python@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /usr/lib/x86_64-linux-gnu/
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY pi/package.json pi/package-lock.json /opt/pi/
RUN cd /opt/pi && node /usr/local/lib/node_modules/npm/bin/npm-cli.js ci --omit=dev --ignore-scripts --no-audit --no-fund \
 && ln -s /opt/pi/node_modules/.bin/pi /usr/local/bin/pi \
 && test "$(pi --version)" = 1.5.2 \
 && mkdir -p /home/agent /work /logs /run/carl \
 && touch /run/carl/capture.sock \
 && printf '%s\n' 'agent:x:1000:1000::/home/agent:/bin/bash' >> /etc/passwd \
 && printf '%s\n' 'agent:x:1000:' >> /etc/group \
 && chown 1000:1000 /home/agent /work /logs
COPY relay.js /usr/local/lib/carl-relay.js
ENV HOME=/home/agent
WORKDIR /work
