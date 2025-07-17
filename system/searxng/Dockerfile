FROM searxng/searxng

COPY ./searxng/requirements.txt /u01/mcp/requirements.txt
RUN /usr/local/searxng/venv/bin/pip install -r /u01/mcp/requirements.txt

COPY ./searxng/mcp_search.py /u01/mcp/mcp_search.py
COPY ./searxng/entrypoint.sh /usr/local/searxng/entrypoint.sh
RUN  chmod 777 /usr/local/searxng/entrypoint.sh
COPY ./searxng/settings.yml /u01/searxng/settings.yml
CMD mv /u01/searxng/settings.yml /etc/searxng/settings.yml

ENTRYPOINT /usr/local/searxng/entrypoint.sh