# Diffusion Models from Scratch

Documento di studio basato sulla lezione del corso Hugging Face: [Diffusion Models from Scratch](https://huggingface.co/learn/diffusion-course/en/unit1/3).

Questo file non riporta il testo originale della pagina, ma una versione rielaborata, completa e pratica dei concetti principali.

## Obiettivo della lezione

La lezione mostra come costruire un modello di diffusione molto semplice da zero, per capire i meccanismi base prima di passare a implementazioni più complete come quelle di `diffusers`.

L’idea centrale è:

- partire da immagini pulite
- corromperle gradualmente con rumore
- addestrare una rete a ricostruire l’immagine originale oppure, nel caso DDPM, a predire il rumore
- usare il modello in modo iterativo per generare campioni nuovi

Il materiale è pensato come un deep dive didattico: utile per capire, meno adatto come implementazione finale da riusare senza modifiche.

## Prerequisiti e setup

La lezione usa librerie standard del mondo PyTorch e Diffusers:

- `torch`
- `torchvision`
- `diffusers`
- `matplotlib`

Il dispositivo viene scelto in modo dinamico, usando GPU se disponibile e CPU altrimenti.

Il notebook usa un dataset piccolo e semplice, tipicamente MNIST, con FashionMNIST come variante plug-and-play un po’ più difficile.

## Dataset usato

L’esempio iniziale lavora con immagini in scala di grigi 28x28.

Motivi della scelta:

- dataset piccolo e rapido da scaricare
- immagini facili da visualizzare
- problema abbastanza semplice da far emergere il comportamento del modello

Ogni sample ha valori normalizzati nell’intervallo da 0 a 1.

## Processo di corruzione

Il primo blocco concettuale è il modo in cui aggiungere rumore ai dati.

La versione toy del notebook usa una miscela lineare tra immagine originale e rumore casuale:

```text
noisy_x = (1 - amount) * x + amount * noise
```

Dove:

- `amount = 0` restituisce l’immagine pulita
- `amount = 1` restituisce rumore puro
- valori intermedi producono livelli diversi di corruzione

Il codice usa broadcasting per applicare un diverso livello di rumore a ogni immagine del batch.

L’obiettivo pedagogico è capire visivamente come la qualità dell’immagine degradi in modo controllato.

## UNet minimale

Per risolvere il problema, la lezione introduce una rete tipo UNet molto piccola.

Caratteristiche del modello minimale:

- input monocolore 28x28
- tre layer convoluzionali nella parte di discesa
- tre layer convoluzionali nella parte di risalita
- skip connection tra blocco down e up
- downsampling con max pooling
- upsampling con `nn.Upsample`

L’idea chiave della UNet è preservare sia contesto globale sia dettagli locali, grazie alle connessioni di skip.

La rete mantiene la stessa shape in output rispetto all’input, così può imparare a restituire un’immagine pulita con la stessa risoluzione.

## Obiettivo di training

Nel toy example il modello riceve in input una versione corrotta e deve predire l’immagine pulita originale.

La loss usata è la mean squared error tra output del modello e immagine pulita.

Il loop di training segue questa sequenza:

1. prendi un batch dal dataloader
2. corrompi ogni immagine con un livello di rumore casuale
3. fai la forward pass del modello
4. calcola la MSE rispetto all’immagine pulita
5. fai backward e optimizer step

Questa formulazione è semplice e didatticamente utile perché fa vedere il problema come denoising diretto.

## Osservazione dei risultati

La lezione mostra che:

- a rumore basso il modello riesce a ricostruire abbastanza bene
- a rumore alto il segnale utile cala e la predizione tende a diventare più sfocata
- il comportamento migliora con più epoche, tuning della rete e dataset più ricco

Questo è un punto importante: il modello può apparire funzionante anche quando non è ancora un buon generatore.

## Sampling nel toy model

Per generare immagini nuove, il notebook non prova a fare tutto in un singolo passaggio.

Invece:

- si parte da rumore casuale
- si chiede al modello una stima dell’immagine denoised
- si avanza solo parzialmente verso quella stima
- si ripete il processo più volte

Questa strategia produce una progressiva emersione di struttura dall’input rumoroso.

Il messaggio pratico è che il sampling iterativo è la parte che rende la diffusione diversa da un semplice autoencoder o denoiser one-shot.

## Versione con diffusers `UNet2DModel`

Dopo il toy model, la lezione sostituisce la rete minimale con `UNet2DModel` di `diffusers`.

Le principali differenze rispetto alla UNet minimale sono:

- GroupNorm nei blocchi
- dropout
- più resnet layer per blocco
- attention in alcuni blocchi
- conditioning sul timestep
- downsampling e upsampling più ricchi e learnable

Questo modello ha molti più parametri e rappresenta una versione più realistica dell’architettura usata nella pratica.

## Corruzione secondo DDPM

La seconda parte della lezione confronta il toy model con l’approccio DDPM.

Nel DDPM il rumore non viene aggiunto con una miscela lineare unica, ma tramite uno schedule di timestep.

L’idea è che a ogni passo si aggiunga una piccola quantità di rumore seguendo una distribuzione gaussiana controllata da una variabile `beta_t`.

Invece di costruire il processo passo per passo ogni volta, si usa una formula chiusa per ottenere direttamente `x_t` da `x_0`.

Questo rende il training più efficiente e allineato alla formulazione teorica del paper.

## Obiettivo di training in DDPM

Nel toy model il target è l’immagine pulita.

Nel DDPM classico il target è il rumore aggiunto.

In pratica:

- si campiona un timestep casuale
- si corrompe l’immagine con il rumore corrispondente a quel timestep
- il modello prova a predire il rumore, non l’immagine originale
- la loss confronta la predizione con il rumore reale

La lezione evidenzia che questa scelta non è solo matematica: cambia anche il peso implicito assegnato ai diversi livelli di rumore durante il training.

## Timestep conditioning

Nel modello `UNet2DModel` il timestep entra come input esplicito.

Questo significa che il modello sa a che livello di rumore sta lavorando.

I vantaggi principali sono:

- migliore capacità di adattarsi a noise level diversi
- comportamento più stabile in training
- allineamento con l’impostazione dei moderni modelli di diffusione

Il notebook sottolinea che si può anche provare a lavorare senza timestep conditioning, ma l’approccio condizionato è generalmente preferito.

## Sampling in DDPM

Anche qui la generazione avviene in più passi.

La differenza è che si usano schedulers e regole più formali per aggiornare il sample ad ogni timestep.

Il sampling moderno lascia aperte varie scelte:

- quanto grande deve essere ogni step
- se usare solo la predizione corrente o anche informazioni storiche
- se aggiungere rumore extra durante il campionamento
- se privilegiare un approccio deterministico o stocastico

La lezione non entra troppo nei dettagli matematici dei sampler, ma chiarisce bene le domande di design che li distinguono.

## Confronto tra toy model e DDPM

Il notebook mette a confronto i due mondi per far emergere le differenze concettuali.

### Toy model

- corruzione semplice e lineare
- target = immagine pulita
- sampling con aggiornamento approssimato
- ottimo per capire i meccanismi di base

### DDPM / diffusers

- schedule di rumore più rigoroso
- target = rumore
- timestep conditioning
- architettura UNet più completa
- sampler più sofisticati

La conclusione è che il toy model aiuta a capire il principio, mentre DDPM rappresenta la versione utile per sistemi reali.

## Cosa imparare dalla lezione

I punti più importanti da portare via sono:

- la diffusione è un problema di denoising iterativo
- controllare il livello di rumore è fondamentale
- la UNet è centrale perché combina contesto e dettagli
- la scelta del target di training cambia il comportamento del modello
- il sampling è parte integrante del modello, non solo un dettaglio di inferenza

## Rilevanza per un modello personale

Per un progetto personale, questa lezione è utile come base concettuale prima di passare a training su immagini reali.

Il flusso consigliato è:

1. capire bene la corruzione e il denoising
2. implementare un esempio piccolo e ripetibile
3. confrontare toy model e `UNet2DModel`
4. solo dopo trasferire le idee a dataset personali o a pipeline con LoRA

In altre parole, la lezione è un ponte tra teoria e pratica.

## Note finali

Il notebook è intenzionalmente educativo e semplificato.

Non è pensato come implementazione finale da copiare così com’è, ma come guida per capire:

- che cosa succede quando si aggiunge rumore
- come una UNet può denoise un’immagine
- perché il timestep conditioning è importante
- perché il sampling richiede più passaggi

Per chi vuole andare oltre, il testo consiglia di esplorare i sampler della libreria `diffusers` e di leggere lavori più avanzati sul design space delle diffusion models.