from fasthtml.common import *

app, rt = fast_app()

@rt("/")
def get():
    return Titled("SentinelGraph Dashboard", P("Il frontend è collegato!"))

serve()