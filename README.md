# Personal Image Studio

Personal Image Studio è la tua base personale per creare immagini con una AI locale, personalizzabile e guidabile anche da agenti esterni. Il progetto non copia Fooocus: si ispira al suo approccio semplice e diretto, ma lo traduce in un'architettura originale, modulare e controllabile da te.

L'idea è questa:

- usi l'app tu, in locale, con la tua interfaccia
- un'altra AI o uno script possono comandarla via API
- puoi cambiare provider, profili e comportamento senza riscrivere tutto

## Cosa fa il progetto

- genera immagini di prova subito, anche senza GPU o modelli pesanti
- espone una UI web locale semplice da usare
- salva la configurazione in un file locale persistente
- permette di creare profili personalizzati
- espone endpoint REST per automazioni e integrazioni future
- lascia il provider di generazione separato dal resto dell'app

## Architettura

Il progetto è diviso in tre livelli:

1. UI web locale: una pagina semplice per prompt, profili e output.
2. API FastAPI: endpoint per generare immagini, leggere la configurazione e gestire i profili.
3. Provider immagine: componente sostituibile che oggi produce anteprime SVG, ma domani può collegarsi a Fooocus o a un altro motore.

## Requisiti

- Windows, Linux o macOS
- Python 3.10 o superiore
- un ambiente virtuale Python consigliato
- opzionalmente una GPU e un backend immagine reale per la fase successiva

## Struttura del progetto

- `src/personal_image_studio/app.py`: applicazione FastAPI, UI e provider
- `pyproject.toml`: dipendenze e configurazione del pacchetto
- `README.md`: documentazione del progetto
- `studio_config.json`: configurazione locale generata a runtime

## Avvio rapido

1. Apri la cartella del progetto.
2. Attiva l'ambiente virtuale Python se vuoi usare quello già presente.
3. Installa il progetto in modalità editabile:

```bash
pip install -e .
```

4. Avvia il server:

```bash
uvicorn personal_image_studio.app:app --reload
```

5. Apri il browser su:

```text
http://127.0.0.1:8000
```

## Come si usa

Nella UI puoi:

- scrivere un prompt
- scegliere un profilo
- impostare stile, aspect ratio e numero di immagini
- aggiungere una nota creativa per automazioni esterne
- generare anteprime immediate

Le anteprime attuali sono SVG dimostrative. Servono per testare il flusso end-to-end senza dipendere ancora da modelli pesanti.

## Profili personalizzati

I profili servono per salvare preset di lavoro riutilizzabili. Il progetto include profili base come:

- default
- cinematic
- portrait
- batch

Puoi modificarli tramite l'API o direttamente nel file `studio_config.json`.

Ogni profilo può controllare:

- nome visualizzato
- stile predefinito
- aspect ratio predefinito
- numero di immagini predefinito
- nota descrittiva

## Configurazione locale

La configurazione persistente viene salvata in `studio_config.json` nella root del progetto. Se il file non esiste, l'app parte con valori di default e lo crea quando serve.

Campi principali:

- `provider`: provider attivo, ad esempio `mock` o `fooocus`
- `fooocus_endpoint`: indirizzo del servizio Fooocus se decidi di collegarlo
- `profiles`: mappa dei profili personalizzati

## Variabili d'ambiente

Puoi sovrascrivere il comportamento del progetto con queste variabili:

- `IMAGE_AI_PROVIDER`: forza il provider attivo, ad esempio `mock` o `fooocus`
- `FOOOCUS_ENDPOINT`: endpoint del backend Fooocus locale o remoto

## Endpoint API

Il progetto è pensato per essere comandato anche da un'altra AI o da uno script. Gli endpoint principali sono:

- `GET /api/health`: stato del servizio e provider attivo
- `GET /api/capabilities`: elenco delle capacità esposte
- `GET /api/config`: configurazione corrente
- `PUT /api/config`: aggiorna configurazione e profili
- `GET /api/profiles`: lista dei profili disponibili
- `POST /api/generate`: genera immagini a partire da un prompt

### Esempio di richiesta di generazione

```json
{
	"prompt": "ritratto cinematografico di un androide in una città al tramonto",
	"negative_prompt": "blurry, low quality, extra fingers",
	"profile": "cinematic",
	"style": "Default",
	"aspect_ratio": "1024x1024",
	"count": 1,
	"seed": null,
	"creative_note": "mantieni palette fredda e luci neon"
}
```

### Esempio di risposta

```json
{
	"provider": "mock",
	"profile": "cinematic",
	"items": [
		{
			"title": "Preview 1",
			"image_data_uri": "data:image/svg+xml;base64,...",
			"prompt": "ritratto cinematografico di un androide in una città al tramonto",
			"notes": "Profilo: cinematic | Style: Cinematic | Aspect ratio: 1344x768 | Nota creativa: mantieni palette fredda e luci neon"
		}
	]
}
```

## Come personalizzare davvero il progetto

Hai tre livelli di personalizzazione:

1. UI: puoi cambiare campi, profili e testo mostrati nell'interfaccia.
2. Configurazione: puoi modificare `studio_config.json` o usare l'endpoint `PUT /api/config`.
3. Motore: puoi sostituire il provider mock con un bridge vero verso Fooocus o un altro servizio.

Se vuoi usare Fooocus più avanti, il punto da collegare è il provider `FooocusBridgeProvider` in `src/personal_image_studio/app.py`.

## Uso con un'altra AI o automazione

Il progetto è pensato per lavorare bene anche con un agente esterno. Un'altra AI può:

- leggere `GET /api/capabilities`
- leggere i profili con `GET /api/profiles`
- aggiornare i profili con `PUT /api/config`
- inviare prompt e note creative con `POST /api/generate`

Questo rende il progetto utile sia per uso manuale sia per workflow automatici.

## Pubblicazione su GitHub

Quando vuoi pubblicarlo sul tuo GitHub personale:

1. crea un repository vuoto sul tuo account GitHub
2. collega il remote al repository locale
3. fai il primo commit
4. fai il push del branch principale

Comandi tipici:

```bash
git add .
git commit -m "Initial personal image studio"
git branch -M main
git remote add origin <URL-del-tuo-repository>
git push -u origin main
```

## Roadmap consigliata

- collegare il provider a Fooocus o a un altro motore reale
- aggiungere cronologia dei job e salvataggio immagini
- creare un editor dei profili direttamente nella UI
- aggiungere esportazione/importazione della configurazione
- preparare una modalità per controllo remoto con chiavi API

## Nota su Fooocus

Fooocus è un progetto esterno noto per il flusso semplice e per la generazione di immagini offline di buona qualità. Questo repository ne richiama l'approccio, ma resta un progetto separato e originale.

## Stato attuale

In questo momento il progetto è già usabile come base locale e come API dimostrativa. La parte di generazione reale può essere collegata in un secondo momento senza cambiare la struttura generale.

## Licenza

GPL-3.0-only.
