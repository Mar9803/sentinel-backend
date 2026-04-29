from fasthtml.common import *
import httpx

hdrs = (Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),)
app, rt = fast_app(hdrs=hdrs)

@rt("/")
def get():
    return Titled("🛡️ SentinelGraph Control Room",
        Main(cls="container mx-auto py-10 px-4")( # Usiamo la doppia parentesi per separare attributi da contenuto
            # Sezione Header
            Div(cls="mb-10 text-center")(
                H1("Monitoraggio Anomalie", cls="text-4xl font-bold text-gray-800"),
                P("Carica un file CSV per analizzare le transazioni attraverso il motore ML+Graph", cls="text-gray-600 mt-2")
            ),
            
            # Form di caricamento
            Div(cls="max-w-xl mx-auto bg-white p-8 rounded-xl shadow-md border border-gray-100")(
                Form(hx_post="/upload", hx_target="#results", hx_encoding="multipart/form-data")(
                    Input(type="file", name="file", cls="block w-full text-sm text-gray-500"),
                    Button("Lancia Analisi", cls="w-full mt-6 bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700 transition duration-200")
                )
            ),

            # Area Risultati
            Div(id="results", cls="mt-12")
        )
    )

@rt("/upload")
async def post(file: UploadFile):
    # 1. Recuperiamo il file caricato
    content = await file.read()
    
    # 2. Inviamo il file al Backend
    async with httpx.AsyncClient() as client:
        files = {'file': (file.filename, content, 'text/csv')}
        try:
            r = await client.post("http://sentinel_backend:8000/analyze", files=files, timeout=30.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return Div(cls="text-red-500 font-bold")(f"❌ Errore di connessione al backend: {e}")

    # --- INIZIO LOGICA SEMAFORO ---
    found = data['anomalies_found']
    
    if found == 0:
        status_cls = "bg-green-100 text-green-800 border-green-200"
        status_text = "✅ SISTEMA SICURO: Nessuna anomalia rilevata."
    elif found < 5:
        status_cls = "bg-yellow-100 text-yellow-800 border-yellow-200"
        status_text = "⚠️ ATTENZIONE: Rilevate attività sospette di bassa entità."
    else:
        status_cls = "bg-red-100 text-red-800 border-red-200"
        status_text = "🚨 PERICOLO: Alto tasso di anomalie rilevato!"
    # --- FINE LOGICA SEMAFORO ---

    # Creazione righe tabella
    rows = [Tr(
                Td(f"{row['pagerank']:.6f}", cls="border px-4 py-2"), 
                Td(f"{row['clustering']:.6f}", cls="border px-4 py-2"), 
                Td(f"{row['anomaly_score']:.4f}", cls="border px-4 py-2 text-red-600 font-medium")
            ) for row in data['results']]

    # Ritorno dei risultati con il widget del rischio in cima
    return Div()(
        # Widget Semaforo
        Div(cls=f"p-4 mb-6 rounded-xl border font-bold text-center {status_cls}")(status_text),
        
        # Titolo Statistiche
        H2(f"Analisi Completata: {found} anomalie su {data['total_analyzed']}", 
           cls="text-2xl font-semibold mb-4 text-gray-700"),
        
        # Tabella Risultati
        Table(cls="min-w-full bg-white border border-gray-200 rounded-lg overflow-hidden")(
            Thead(cls="bg-gray-50")(
                Tr(
                    Th("PageRank", cls="px-4 py-2"), 
                    Th("Clustering", cls="px-4 py-2"), 
                    Th("Anomaly Score", cls="px-4 py-2")
                )
            ),
            Tbody()(*rows)
        )
    )

serve()