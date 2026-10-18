# Real-Time-GNN-based-Card-Not-Present-fraud-alerting-prototype
Account Fraud compromise alerts from CNP transactions during travel.


A Django app that trains four fraud models on any labelled card-transaction CSV and compares them. The four models are XGBoost, a stacked ensemble, a graph neural network and an LSTM-CNN sequence model. The app then replays unseen transactions through them in real time and raises alerts that explain themselves with behavioural signals.

## Quick start

```bash
cd khumo-detection-fraud-systems
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt 
python manage.py migrate
python manage.py createsuperuser      # your analyst login
python manage.py runserver
```

The analyst console at http://127.0.0.1:8000 needs a staff login. The customer portal lives at http://127.0.0.1:8000/my/.

Open http://127.0.0.1:8000, choose **Upload**, and drop in `creditcard_cnp_20k-1.csv`. Training takes about a minute for 20,000 rows. After that the dataset page shows the results, and **Open live monitor** starts the stream.

For big files you can train from the command line instead:

```bash
python manage.py train_dataset path/to/file.csv --name "My dataset"
```

## Pages

| Page | What it does |
|---|---|
| Datasets `/` | Every upload, its status and its best model. |
| Upload `/upload/` | Upload a CSV. Columns are detected automatically and training starts in a background thread. |
| Results `/datasets/<id>/` | Shows ROC-AUC, PR-AUC, F1, accuracy, precision and recall per model, plus confusion matrices (TP, FP, TN, FN), ROC curves, XGBoost feature importance and a behavioural-signal lift table. Each dataset has its own page. |
| Results `/results/` | One page for comparing the models. It has a scorecard per model showing ROC-AUC, PR-AUC, F1, accuracy, precision, recall, TP/FP/FN/TN and threshold. A head-to-head dot plot shows every metric on a zoomed axis. Charts follow: ROC and precision–recall curves with each model's operating point, frauds caught vs missed, false alarms, score distributions per model and training curves. Every chart has hover values, a legend that hides a model on all charts at once, and a data table. Then the whole **transaction graph** (see below). Below that, every trained dataset is listed side by side with an F1-by-dataset chart and a leaderboard. **Download CSV** exports all of it. |
| Live monitor `/monitor/<id>/` | Streams the test period through the live engine over Server-Sent Events. Alerts appear as they fire. |
| Alert explanation `/alerts/<id>/` | Why a transaction was flagged: a plain-language summary, each model's score against its threshold, feature attributions per model, exact XGBoost SHAP values, graph evidence from the GNN and a counterfactual. |
| Customers `/customers/` | Enrol customers in the portal and see how many of their transactions were flagged and how many alerts they have replied to. |
| Alerts `/alerts/` | Review queue with "Confirm fraud" and "False alarm" buttons. New alerts also pop up on every page and as desktop notifications. |

## How it works

**Split.** Every row of the file is used; nothing is sampled away. Rows are sorted by time. The first 60% trains the models. The next 15% is used for early stopping and for choosing each model's alert threshold (the one that maximises F1). The last 25% is the test set, and every number on the results page comes from it. The live monitor replays the whole test period. If the later slices hold fewer than 5 frauds, a stratified split is used instead and the page says so.

**Class imbalance: FSMOTE** (`detector/ml/fsmote.py`). Fraud is usually well under 1% of transactions. Instead of sampling legitimate rows away, the training slice gets synthetic frauds from Fuzzy-SMOTE (Xu et al., 2017, *Int. J. Distributed Sensor Networks* 13(4), doi:10.1177/1550147717703116):

1. Each training fraud gets a fuzzy membership to the fraud class from its distances to the fraud and legitimate centroids (fuzzy c-means formula, m = 2, on standardised features).
2. Frauds with membership ≤ 0.7, the ones that look most like legitimate transactions, are the seeds. Each gets a share of the synthetic budget proportional to (1 − μ)/μ, so harder frauds get more neighbours.
3. Each synthetic fraud is x + λ·(neighbour − x) for one of the seed's 5 nearest fraud neighbours and λ ~ U(0, 1).
4. One-hot categories and 0/1 flags are copied from the nearer parent rather than averaged (as in SMOTE-NC). Whole-number columns such as transaction counts are rounded, so no synthetic row is recognisably synthetic.

The budget fills training up to the `FSMOTE_RATIO` frauds-to-legitimate ratio (default `1.0`, fully balanced). Set it to e.g. `0.5` for half, or `0` to switch FSMOTE off and fall back to class weights; that is useful as a baseline in a write-up. Any imbalance left after oversampling is handled with class weights. **Validation and test data are never oversampled**, so thresholds and every reported metric come from real transactions only.

All four models get the same synthetic frauds, each in its own input space:

- **XGBoost and the ensemble** train on the feature rows.
- **GNN:** a synthetic node is the same λ-mix of its two parents' inputs *and* graph neighbourhoods, in the style of GraphSMOTE (Zhao et al., 2021). It only receives messages, so real nodes are unaffected. The gradients are hand-derived and gradient-checked.
- **LSTM-CNN:** each synthetic sequence mixes its parents' recent-transaction windows step by step, and its last step is the synthetic feature row.

The results and dataset pages show how many synthetic frauds were added and from how many seeds.

**Behavioural features** (`detector/ml/behavior.py`). One `BehaviorState` processes transactions in time order. It keeps per-account profiles:

- amount mean and standard deviation, time since the last transaction
- 1-hour and 24-hour velocity
- devices, IPs, merchants and countries seen before

It also tracks how many accounts share each device and IP. Features only ever use an account's past. The same code runs at training time and at live scoring time, so the two cannot drift apart. The rules built on top of these features become the plain-language reasons shown on alerts, for example "Transaction in NG; this account normally transacts in ZA".

**Models** (`detector/ml/pipeline.py`):

- **XGBoost**: gradient-boosted trees with `scale_pos_weight` and early stopping on validation PR-AUC.
- **Ensemble**: a `StackingClassifier` of Random Forest, Extra Trees and scaled Logistic Regression, with a logistic-regression meta-learner. On very large training sets each forest tree bootstraps 60,000 rows, so training time and model size stay flat while every row is still used by some trees.
- **GNN** (`detector/ml/gnn.py`): a two-layer GraphSAGE-style network.
  - Every transaction is a node, linked through the account, card, device, IP, merchant and terminal it shares with other transactions. Each entity type has its own weights.
  - It is written in NumPy and SciPy with hand-derived gradients and Adam, so PyTorch isn't needed.
  - It is inductive: a new live transaction is scored from stored per-entity embeddings, which are then updated.
- **LSTM-CNN** (`detector/ml/lstm_cnn.py`): a sequence model over each account's last 8 transactions.
  - Each step is the same feature vector the other models use, plus a "real transaction" flag; shorter histories are zero-padded.
  - A causal 1-D convolution (kernel 3, 32 filters) picks up local patterns between neighbouring transactions, an LSTM (32 units) reads the whole window in order, and a logistic output uses the LSTM's final state plus the current step's convolution features.
  - Weighted cross-entropy for the class imbalance, Adam with gradient clipping, early stopping on validation PR-AUC.
  - Like the GNN it is NumPy with hand-derived (gradient-checked) backpropagation, so PyTorch and TensorFlow aren't needed. It trains in a few seconds on 20,000 rows.
  - Sequences follow the account column, or the card column if there is no account. Without either, each transaction is read on its own.
  - Live scoring keeps each account's last 7 feature vectors and appends every new transaction after scoring it.
- **Combined**: the average of the four probabilities. This is what the live monitor alerts on.

**Alert levels.**

- *Likely fraud*: the Combined score passes its threshold, or at least two of the four models flag the transaction.
- *Needs review*: exactly one model flags it, or its behavioural signal weight is 5 or more while the Combined score is at least a third of its threshold.

## Transaction graph

The Results page draws the **whole** graph the GNN learns from (`detector/ml/txgraph.py`, `static/detector/graph.js`). Every transaction is a node linked to its account, card, device, IP address and merchant (and terminal, if the file has one). Fraudulent transactions are red, and entities that took part in fraud have a red ring.

- **Layout** is computed once at training time. Each account sits in the middle of its own cluster, with its cards, devices and IPs around it, and its transactions as a cloud of dots. Clusters are packed largest-first on a spiral. Clusters joined through shared devices or IPs are laid out by force. Very large ones are first coarsened (private devices and IPs orbit their only account), then laid out with a particle-mesh force layout. Merchants link almost every account, so they sit on an outer ring.
- **Fraud rings** are found on the fraud-only subgraph: accounts whose *fraudulent* transactions share a device, IP or card. Each is listed with the shared entities as evidence and zooms to them when clicked.
- **Exploring:** drag to pan; scroll or pinch to zoom. Hover any node for a summary. Click one to trace everything within two hops; the panel shows its details, and for a transaction its ID, amount, time, label and GNN score. You can also search for any account, card, device, IP or merchant, and toggle entity types. Keyboard: arrows pan, + and − zoom, 0 fits, Esc clears.
- **Scale:** a canvas renderer with culling and a cached bitmap while you move, so it stays usable at hundreds of thousands of transactions. On files over 60,000 transactions, merchant links start hidden because nearly every account shares them. Tick the box to draw them.
- **Download the edge list (CSV)** from the graph section (`/datasets/<id>/graph/edges.csv`) to open the same graph in Gephi or NetworkX.

## Customer portal (`/my/`)

A separate site where each customer watches **only their own account**.

- **Live activity.** The page checks for new transactions every 3 seconds. Every transaction on the account appears the moment the live monitor or the scoring API processes it, marked *Normal*, *Being checked* or *Flagged as suspicious*.
- **Real-time alerts.** A flagged transaction triggers an on-screen notice, a browser notification (if the customer allows them) and an unread count in the tab title. It also sends an email to the customer if they have an address saved and email alerts turned on.
- **"Did you make this transaction?"** Each flagged transaction has two buttons. The customer's answer ("it was me" or "not me") appears on the analyst's alert list and alert page.
- **Customer-safe reasons.** Customers see the behavioural reasons in plain words, such as "Transaction in NG; your account normally transacts in ZA". They never see model scores, feature attributions or SHAP values, because telling a fraudster which signals the models react to would help them avoid detection.

**Enrolling customers:**

- *Self-service* at `/my/signup/`. The account number is checked against the last four digits of a card on that account; this uses the `card_last4` column in the dataset. The error message is the same whether the account or the card is wrong, so the form can't be used to discover which account numbers exist.
- *By an analyst* on the **Customers** page.
- *From the command line:*
  ```bash
  python manage.py create_customer --dataset 1 --account ACC0034401 --username michael --password 'Str0ng-pass-99' --email you@example.com
  ```

Only enrolled accounts are recorded, and each customer's recent earlier transactions are copied in when they enrol.

**Demo.** Enrol `ACC0034401`; its card ends in `3700`. Sign in at `/my/` in one browser window. In the analyst window, open the Live monitor, set *Start at transaction* to `1900` and start the stream. Four fraudulent transactions from Nigeria reach the customer at transactions 3,919–3,937.

**Access rules:**
- Customers can't open analyst pages; they are sent back to `/my/`.
- Customer logins are refused at the analyst sign-in, and staff logins at the customer sign-in.
- A customer can only reply to alerts on their own account. For anyone else's alert they get "not found", which doesn't reveal that it exists.
- Set `ANALYST_LOGIN_REQUIRED=0` to switch off the staff login for local experiments.
- Set `SCORING_API_KEY` to require an `X-Api-Key` header on the scoring API.

## Explainable AI (XAI)

Every flagged transaction gets an explanation at the moment it's scored (`detector/ml/explain.py`). It's stored on the alert and shown on its page, in the live-monitor card, in the email and in the API response. The explanation has five parts:

- **Plain-language summary.** Built from the strongest drivers, for example: "Flagged mainly because V14 is lower than usual (-5.71 vs -0.03) and the device has never been used on this account. Changing … would lower the combined score from 93% to 30%, below the 34% alert threshold."
- **Feature attribution for every model.** Each feature group is swapped for its value in 12 legitimate training transactions, and the average drop in fraud probability is recorded. A feature group is a numeric column, a whole categorical column or one behavioural feature. This works the same way for XGBoost, the stacked ensemble, the GNN and the LSTM-CNN (for the LSTM-CNN only the flagged transaction is swapped; the account's earlier transactions stay as they were). The Combined view is the mean of the four, matching how the Combined score is calculated.
- **Exact XGBoost SHAP.** TreeSHAP values from `pred_contribs`, in log-odds, grouped the same way as the attributions.
- **Graph evidence.** The GNN rescores the transaction with each linked entity (account, card, device, IP, merchant, terminal) hidden in turn. This shows which connections raised or lowered the risk, together with how many earlier transactions and accounts share that entity.
- **Counterfactual.** The top drivers are reset one more at a time until the Combined score falls below its alert threshold.

Explanations add about 80 ms per flagged transaction; transactions that aren't flagged aren't explained. Retrain datasets that were trained before this feature so their models include the background sample. Older models still work but compare against a single median transaction.

Limits:
- Occlusion shares credit between correlated features.
- A model that is already certain (99% or more) barely moves when one input changes.
- The V-columns are anonymised, so they can be ranked but not described.

## Email alerts

By default, emails are printed to the console. To send real mail:

```bash
export ALERT_EMAIL_TO=you@example.com
export EMAIL_HOST=smtp.gmail.com EMAIL_PORT=587 EMAIL_USE_TLS=1
export EMAIL_HOST_USER=you@gmail.com EMAIL_HOST_PASSWORD=app-password
export ALERT_EMAIL_LEVELS=fraud          # or fraud,review
```

## Scoring API

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/1/score/ \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"transaction_id": "T1", "timestamp": "2013-09-25 03:12:00",
       "account_id": "ACC0012700", "Amount": 4999, "tx_country": "NG",
       "device_id": "DEVnew", "channel": "ECOM"}}'
```

Send `{"transactions": [...]}` to score a batch. Use the dataset's own column names. Missing numeric values fall back to training medians and missing categories to the most common value. Each call updates the account's history, so send transactions in time order.

## What a dataset needs

- A 0/1 label column named `Class`, `isFraud`, `is_fraud`, `fraud`, `label` or `target`.
- Some feature columns.
- At least 20 frauds.
- Optionally, a timestamp and entity IDs (`account_id`, `card_token`, `device_id`, `ip_address`, `merchant_id`, `terminal_id`) for the behavioural signals and the graph. Other common names are recognised too; see `detector/ml/schema.py`.

Files without real labels are rejected with an explanation. The IEEE-CIS `sample_submission.csv` is one example: it contains only an ID and `isFraud = 0.5` on every row. Every row is used for training, however large the file.

## Branding

The logo and favicon live in `detector/static/detector/img/`. The brand palette is defined once, as CSS variables at the top of `detector/static/detector/app.css`:

| Colour | Hex | Used for |
|---|---|---|
| Alarm red | `#C40C0C` | Fraud alerts, mistakes in confusion matrices, the GNN series |
| Signal orange | `#FF6500` | Review alerts, active navigation, focus rings, the Ensemble series |
| Sequence blue | `#2A78D6` (`#4D9DE8` in dark mode) | The LSTM-CNN series. The only cool colour, so it stays distinct from the three warm model colours |
| Elephant rust | `#CC561E` | Button hover, feature bars, the XGBoost series |
| Savanna gold | `#F6CE71` | Correct decisions in confusion matrices, risk meters, review backgrounds |

The Django project package is `kdfs`, and the app is `detector`.

## Notes and limits

- Datasets trained before FSMOTE and the transaction graph were added keep working. Retrain them to add both.
- Full-size files take a while: in testing, a 285,000-row file (ULB size) trained in about 13 minutes on a 2-core machine, with memory peaking around 3 GB. The ensemble and the GNN take most of that time, and more cores speed up the forests. Use `python manage.py train_dataset` for big files so the browser isn't involved.
- SMOTE-family oversampling doesn't always help every model; on some data a model scores better with class weights alone. Train once with the default and once with `FSMOTE_RATIO=0` to compare on your own data.

- Datasets trained before the LSTM-CNN was added keep working with their original three models. Retrain them to add the LSTM-CNN and the precision–recall, score-distribution and training charts (ROC curves and the caught/false-alarm charts work for every dataset).
- The LSTM-CNN is evaluated the same way as the GNN: each test transaction is scored with the earlier transactions on its account as context, but labels are never used.

- Training runs in a thread inside the web process, which keeps the app free of extra infrastructure. For production, move `tasks.start_training` to Celery or RQ and serve with a real WSGI/ASGI server.
- On `creditcard_cnp_20k-1.csv`, much of the predictive power comes from the ULB PCA columns (V17, V12, V14). Scores on data without those columns will be lower.
- The GNN is evaluated transductively: test nodes are in the graph, but their labels are never used. The live monitor scores the same transactions inductively and produced the same catch rate in testing.

