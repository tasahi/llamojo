# Timestral


## About 

Released in October 2026, Timestral is a family of open-source time-series foundation models developed by Mistral AI. Unlike general-purpose Large Language Models (LLMs) built for text, Timestral belongs to the class of Large Quantitative Models (LQMs). It is specifically engineered to process ordered numerical observations, predict multivariate future sequences, and quantify uncertainty for operations like demand forecasting, financial planning, and supply chain tracking. [1] (https://mistral.ai/news/timestral/), [2] (https://aiweekly.co/alerts/mistral-opens-timestral-209m-time-series-model-beating-timesfm-3), [3] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri), [4] (https://www.datastudios.org/post/mistral-timestral-time-series-forecasting-5m-28m-123m-209m)
Despite its highly compact footprint, the flagship 209-million-parameter model achieved state-of-the-art status by outperforming significantly larger industry benchmarks in zero-shot accuracy. [1] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri), [2] (https://mistral.ai/news/timestral/), [3] (https://aiweekly.co/alerts/mistral-opens-timestral-209m-time-series-model-beating-timesfm-3)

Core Architecture & Innovations

Timestral achieves its forecasting performance and efficiency through three fundamental technical innovations: [1] (https://arxiv.org/html/2610.37585v1)
• Decoupled Horizon Forecasting (DHF): Traditional time-series transformers often couple prediction horizons with the context length, which requires complex masking strategies. DHF separates learned prediction tokens from causally encoded historical contexts. This allows the model to scale to massive context windows (up to 32,768 steps) without masking, rendering every past point as both a target and active context. [1] (https://github.com/mistralai/timestral), [2] (https://www.researchgate.net/publication/417008702_Timestral), [3] (https://docs.mistral.ai/models/timestral-1-0-209m), [4] (https://arxiv.org/html/2610.37585v1)
• Role-Aware Covariate Framework: The framework natively handles known-future covariates (e.g., scheduled holiday markers or weather forecasts) alongside past-only covariates. It feeds future variables into the forecasting sequence while mathematically locking out future target data to strictly preserve target causality. [1] (https://www.researchgate.net/publication/417008702_Timestral), [2] (https://mistral.ai/news/timestral/), [3] (https://github.com/mistralai/timestral), [4] (https://arxiv.org/html/2610.37585v1)
• Probabilistic Quantile Outputs: Instead of generating rigid, single-value deterministic predictions, Timestral outputs comprehensive probability distributions. It estimates 21 distinct quantiles per time step, enabling precise risk management and downside safety analysis. [1] (https://daily.dev/posts/mistral-just-released-the-best-forecasting-model-timestral--fnllusjtu), [2] (https://github.com/mistralai/timestral)

Model Sizes & Performance Metrics

The family spans four scaled architectures to balance computation budgets and accuracy. All versions are trained via a two-stage pipeline consisting of broad synthetic pre-training followed by real-world structured multivariate dependency post-training. [1] (https://mistral.ai/news/timestral/), [2] (https://arxiv.org/html/2610.37585v1), [3] (https://www.researchgate.net/publication/417008702_Timestral)

Model Variant	Parameter Count	File / Checkpoint Size	Target Environment & Latency
Timestral-5M	5 Million	~21 MB	Ultra-low latency, running locally in browsers or smartphones via ONNX (~10ms).
Timestral-28M	28 Million	~111 MB	Lightweight edge computing and fast embedded systems.
Timestral-123M	123 Million	~490 MB	Balanced server-side execution; ranked #2 globally on major benchmarks.
Timestral-209M	209 Million	~837 MB	Flagship variant; ranked #1 globally on public benchmarks (~140ms on CPU).


Benchmark Rankings & Licensing

The largest variant (Timestral-209M) achieved the #1 zero-shot rank across the time-series domain's three core public evaluation suites: GIFT-Eval, TIME (98 tasks), and fev-bench (100 tasks). It successfully beat dominant models like Google’s TimesFM-3 and Amazon's Kronos 2 while relying on roughly two-thirds of their parameters. [1] (https://aiweekly.co/alerts/mistral-opens-timestral-209m-time-series-model-beating-timesfm-3), [2] (https://github.com/mistralai/timestral), [3] (https://mistral.ai/news/timestral/), [4] (https://daily.dev/posts/mistral-just-released-the-best-forecasting-model-timestral--fnllusjtu)
The model is distributed under a split open-source mechanism: [1] (https://github.com/mistralai/timestral)
• Codebase & Architecture: Openly accessible via the Official Mistral Timestral GitHub Repository under the Apache 2.0 License.
• Model Weights & Checkpoints: Hosted directly on Hugging Face under a Modified MIT License. Fine-tuning pipelines are fully supported via the Mistral Forge ecosystem. [1] (https://mistral.ai/news/timestral/), [2] (https://github.com/mistralai/timestral), [3] (https://aiweekly.co/alerts/mistral-opens-timestral-209m-time-series-model-beating-timesfm-3)
Are you planning to deploy Timestral for a specific application? If so, tell me:
• What industry / domain are you forecasting for (e.g., retail demand, energy grid load, financial markets)?
• Do you intend to use known-future covariates (like weather data or calendar promotions)?
• Will your final architecture run on local CPU edge hardware or a GPU server cluster?
I can provide code integration structures or data formatting templates to help you begin.



## Comparison with other forecasters

Comparing Mistral's Timestral against  represents pitting a modern Time-Series Foundation Model (Deep Learning) against a classic Additive Curve-Fitting Statistical Tool. [1] (https://explainx.ai/blog/google-timesfm-3-multivariate-time-series-foundation-model-2026)
Independent industry backtests evaluating both models a year in advance show a staggering accuracy gap: where a baseline "lazy guess" was off by €100, Timestral was off by €17, while Meta's Prophet was off by €21. [1] (https://www.linkedin.com/posts/assemchammah_mistral-just-launched-timestral-just-used-activity-7524055599848980481-vYD_)

Core Structural Differences


Feature	Mistral Timestral	Meta Prophet
Underlying Tech	Causal Patch-Based Transformer.	Generalized Additive Model (Linear trend + Fourier series).
Training Pipeline	Zero-Shot Foundation Model: Pre-trained on billions of sequences; works out of the box without local data training.	Train-from-Scratch: Must be fit and calibrated individually to every single metric.
Data Requirements	Handles very short or massive historical context (up to 32,768 steps).	Requires a minimum of 2–3 years of data to effectively map yearly seasonality.
Output Type	Multi-variate probabilistic forecasting (21 distinct quantiles per step).	Univariate curve forecasting with basic parametric uncertainty intervals.


Where Timestral Wins Hands Down

• Zero-Shot & Multi-Series Scale: Prophet scales poorly if you have thousands of individual metrics (e.g., SKUs or server logs) because you have to train and maintain thousands of separate models. Timestral evaluates multi-series trends seamlessly in a single forward pass without localized training. [1] (https://www.datastudios.org/post/mistral-timestral-time-series-forecasting-5m-28m-123m-209m), [2] (https://encorp.ai/en/blog/ai-analytics-timestral-cpu-forecasting-2026-10-19)
• Complex, Non-Linear Patterns: Prophet is essentially a curve-fitting tool that assumes your data changes in smooth linear trends, cycles, and holiday jumps. It notoriously struggles with volatile short-term fluctuations or irregular macro shifts, which Timestral's deep-learning framework naturally absorbs. [1] (https://www.researchgate.net/publication/387701628_A_Comparative_Study_of_ARIMA_Prophet_and_LSTM_for_Time_Series_Prediction), [2] (https://facebook.github.io/prophet/), [3] (https://encorp.ai/en/blog/ai-analytics-timestral-cpu-forecasting-2026-10-19)
• Multivariate Correlations: Prophet is built primarily for a single variable. Timestral supports complex dependencies via a Role-Aware Covariate Framework, mapping how targets interact with past-only and known-future parameters simultaneously. [1] (https://arxiv.org/html/2610.37585v1), [2] (https://github.com/mistralai/timestral), [3] (https://explainx.ai/blog/google-timesfm-3-multivariate-time-series-foundation-model-2026)

Where Prophet Might Still Have a Niche

• Total Explainability: Prophet decomposes its forecast directly into a readable chart of parts (e.g., "Trend + Yearly Seasonality + Day-of-Week Effect"). If your finance or ops team values manual auditing and extreme transparency over pure accuracy, Prophet's outputs are far easier to dissect than a Transformer's embeddings. [1] (https://explainx.ai/blog/google-timesfm-3-multivariate-time-series-foundation-model-2026), [2] (https://facebook.github.io/prophet/)
• Simplicity and Custom Holiday Locking: If your metric is strictly driven by known calendar schedules (like a retail site that spikes exactly on Black Friday or Easter), Prophet's built-in holiday regressor matrix remains lightweight, lightning-fast to run on local machines, and simple to explicitly hardcode. [1] (https://www.artefact.com/blog/is-facebook-prophet-suited-for-doing-good-predictions-in-a-real-world-project/), [2] (https://explainx.ai/blog/google-timesfm-3-multivariate-time-series-foundation-model-2026), [3] (https://facebook.github.io/prophet/)

The Bottom Line

If you are forecasting a single business metric with distinct, long-running calendar cycles, Prophet remains a decent, cheap, and easily explainable baseline. However, for high-dimensional data, cross-correlated variables, volatile sequences, or sprawling multi-product operations, Timestral provides a completely superior, state-of-the-art alternative that saves engineering hours by bypassing the need for tedious manual model tuning. [1] (https://github.com/mistralai/timestral), [2] (https://explainx.ai/blog/google-timesfm-3-multivariate-time-series-foundation-model-2026), [3] (https://encorp.ai/en/blog/ai-analytics-timestral-cpu-forecasting-2026-10-19), [4] (https://www.datastudios.org/post/mistral-timestral-time-series-forecasting-5m-28m-123m-209m)


## Inputs

When feeding data to Timestral, you do not pass text prompts or raw unformatted timestamps. Instead, you provide ordered numerical series and categorical arrays chunked into specific structural roles. [1] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri), [2] (https://www.qovery.com/blog/self-host-timestral-time-series-forecasting-own-cloud), [3] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/)
Timestral’s Role-Aware Covariate Framework handles data by categorizing inputs into one of three structural types, split by continuous (numeric) or categorical data types. [1] (https://mistral.ai/news/timestral/), [2] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri)

1. The Three Structural Input Roles

To avoid leaking future target information while still benefiting from known schedules, you must strictly assign every data sequence to one of these three fields:
• Target Variables (Historical Context Only): These are the metric(s) you actually want to forecast (e.g., actual daily sales, energy grid load, or sensor telemetry). You provide its historical sequence up to the "forecast origin" (the data cutoff point). [1] (https://arxiv.org/html/2610.37585v1), [2] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/), [3] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri)
• Past-Only Covariates: These are external variables that help explain the target, but you only have their historical records up to the present moment.
	• Examples: Yesterday’s store footfall, a server's memory usage history, or actual historical wind speed. [1] (https://www.nxcode.io/resources/news/timestral-forecasting-known-future-inputs-2026), [2] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri)
• Known-Future Covariates: These are variables where you already know the exact future sequence across the entire forecast window. They must span both your historical context length and the requested future forecast horizon.
	• Examples: A marketing or holiday calendar, an approved production schedule, or an expected weather forecast issued at the cutoff time. [1] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/), [2] (https://www.nxcode.io/resources/news/timestral-forecasting-known-future-inputs-2026), [3] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri)

2. Supported Data Types & Formatting


A. Continuous (Numeric) Inputs

Numerical entries are handled natively by the transformer's patch embeddings. [1] (https://github.com/mistralai/timestral/blob/main/README.md), [2] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/)
• Structure: Fed as multi-dimensional continuous arrays (commonly PyTorch tensors or NumPy arrays).
• Context Padding: Timestral supports massive historical context windows—up to 32,768 time steps. You do not need to apply complex attention masking; the model’s Decoupled Horizon Forecasting (DHF) handles historical separation automatically. [1] (https://explainx.ai/blog/timestral-mistral-time-series-foundation-models-209m-cpu-forecasting-2026), [2] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri), [3] (https://mistral.ai/news/timestral/), [4] (https://arxiv.org/html/2610.37585v1)

B. Categorical Inputs

For data points that lack a natural numerical scale, Timestral accepts them natively as categorical covariates to prevent forcing artificial weights onto arbitrary numbers. [1] (https://mistral.ai/news/timestral/), [2] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/)
• Examples: Day of the week, weather conditions (sunny, rainy), or machine states (running, idle, under maintenance).
• Formatting Requirement: In the official , categories must be mapped to integer numeric codes (e.g., using pandas.factorize).
• Missing Values / Padding: Use NaN for missing categorical values or context padding. The model uses internal target-encoding functions (like timestral.covariate_encoding.target_encode) to process these integer arrays automatically before passing them to the attention heads. [1] (https://mistral.ai/news/timestral/), [2] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/), [3] (https://github.com/mistralai/timestral/blob/main/README.md)

High-Level Summary Checklist for your Code Pipeline:

1. Format targets and covariates as standard continuous arrays.
2. Factorize categorical fields into flat integer sequences.
3. Align Lengths: Ensure your targets and past covariates stop at your cutoff point, while known-future arrays continue all the way through your future prediction window. [1] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri), [2] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/), [3] (https://github.com/mistralai/timestral/blob/main/README.md), [4] (https://www.nxcode.io/resources/news/timestral-forecasting-known-future-inputs-2026)


## Examples of inputs

The reference API layout from the Official Mistral Timestral Repository requires shapes structured around a three-dimensional layout: (Batch, Variates, Time). [1] (https://github.com/mistralai/timestral), [2] (https://github.com/mistralai/timestral/blob/main/README.md)
Here are the concrete formatting examples for feeding data into . [1] (https://github.com/mistralai/timestral)

1. Basic Structure (Univariate or Multivariate Target Only)

If you only have the target history (e.g., historical sales metrics) and no external variables, the model accepts data as a Python array, a NumPy array, or a PyTorch tensor. Missing values must be marked explicitly using np.nan. [1] (https://github.com/mistralai/timestral/blob/main/README.md), [2] (https://github.com/mistralai/timestral)
python
import numpy as np
from timestral import TimestralForecaster

# Initialize the model 
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")

# Shape format: (Batch/Series, Variates, Time)
# Here: 1 series, 1 target variable, 100 historical time steps
context = np.random.randn(1, 1, 100) 

# Injecting missing values using NaN (Timestral handles these via internal indicator masks)
context[0, 0, 15] = np.nan  
context[0, 0, 16] = np.nan  

# Execute zero-shot forecast for a 32-step future window
forecast = forecaster.forecast(context, prediction_length=32)
Use code with caution.

2. Adding Numeric Numeric Covariates (Past-Only & Known-Future)

When passing continuous numerical indicators, you provide them as auxiliary arrays. [1] (https://www.qovery.com/blog/self-host-timestral-time-series-forecasting-own-cloud)
• past_covariates must precisely match the length of your history timeline (T).
• known_covariates must span the historical timeline plus the entire desired prediction length (T + H). [1] (https://github.com/mistralai/timestral/blob/main/README.md)
python
# Setup lengths
T = 100   # Historical context length
H = 32    # Forecast horizon length

# 1. Target Data (History only: shape 1, 1, 100)
context = np.random.randn(1, 1, T)

# 2. Past-Only Covariates (e.g., historical competitor store traffic)
# Shape: (Batch, Num_Past_Covariates, T) -> 1 series, 2 features, 100 steps
past = np.random.randn(1, 2, T)

# 3. Known-Future Covariates (e.g., expected temperature grid shifts)
# Shape: (Batch, Num_Known_Covariates, T + H) -> 1 series, 1 feature, 132 steps
known = np.random.randn(1, 1, T + H)

# Pass them all to the forecaster
forecast = forecaster.forecast(
    context=context,
    prediction_length=H,
    past_covariates=past,
    known_covariates=known
)
Use code with caution.

3. Adding Categorical Covariates

Timestral does not accept raw string categories like "Monday" or "Under Maintenance". String arrays must be mapped to flat, integer numeric codes (using tools like pandas.factorize), mapping missing data directly to np.nan. [1] (https://mistral.ai/news/timestral/), [2] (https://github.com/mistralai/timestral/blob/main/README.md)
python
import pandas as pd

# Example raw historical categorical features
days_history = ["Mon", "Tue", "Wed", "Thu", "Fri"]
days_future  = ["Sat", "Sun"]

# Map strings cleanly to integer category identifiers using pandas
all_days = days_history + days_future
codes, uniques = pd.factorize(all_days)

# Split strings back out into context vs future arrays
# Timestral expects dimensions: (Batch, Num_Categorical_Features, Time)
codes_past  = codes[:5].reshape(1, 1, 5).astype(float)        # Shape: (1, 1, 5)
codes_known = codes.reshape(1, 1, 7).astype(float)            # Shape: (1, 1, 7) - covers history + future

# Target history matching the 5 past steps
context = np.array([[[102.5, 98.4, 115.1, 120.0, 142.3]]])   # Shape: (1, 1, 5)

# Forecast using native categorical injections
forecast = forecaster.forecast(
    context=context,
    prediction_length=2,
    categorical_past_covariates=codes_past,
    categorical_known_covariates=codes_known
)
Use code with caution.

Inspecting Output Tensors

Regardless of your input layout, the output structure returns a probabilistic prediction block containing 21 quantile values per sequence step: [1] (https://ai-tldr.dev/releases/mistral-timestral/), [2] (https://github.com/mistralai/timestral)
python
# Output tensor dimensions match: (Batch, Variates, Horizon, Quantiles)
print(forecast.quantiles.shape)  # Returns: (1, 1, 32, 21)

# Extract your target baseline median or specific risk barriers easily
median_prediction = forecast.median        # Shape: (1, 1, 32)
p90_worst_case    = forecast.quantile(0.9)  # Shape: (1, 1, 32)
Use code with caution.
Would you like assistance in structuring a Pandas data preparation script to turn your raw database timestamps and string labels into these 3D NumPy arrays? If so, share a quick description of your current raw column names.


## Imputation

Yes, Timestral is natively designed to handle and bypass missing values, but because it is a strict causal transformer, you use it for imputation differently than traditional bidirectional models (like MICE or masked autoencoders).
Instead of reconstructing an isolated gap in the middle of a past timeline via bidirectional interpolation, Timestral infers the statistical probability of missing historical points by running its forecasting engine sequentially across the timeline. [1] (https://arxiv.org/html/2610.37585v1), [2] (https://www.researchgate.net/publication/417008702_Timestral)

How Timestral Handles Missing Data Under the Hood

In older foundation models (like TimesFM-3), random masking was used during training to simulate missing data. This meant that if your historical data had a chunk of missing values, the model lost that entire timeframe as active context for later points. [1] (https://arxiv.org/html/2610.37585v1)
Timestral solves this using Decoupled Horizon Forecasting (DHF). [1] (https://arxiv.org/html/2610.37585v1)
1. Explicit NaN Flagging: You pass missing values directly into the history array as np.nan (or null in its JSON API).
2. Context Retention: The DHF architecture separates the true missing observations from its internal prediction mechanics. The model acknowledges that a value is missing at time T, but it still utilizes the broader surrounding structural context to project forward without degrading the timeline. [1] (https://github.com/mistralai/timestral/blob/main/README.md), [2] (https://ai-tldr.dev/releases/mistral-timestral/), [3] (https://localai.io/docs/features/forecast/index.html), [4] (https://www.remio.ai/post/timestral-mistral-release-time-series-foundation-models-that-challenge-bigger-ri), [5] (https://www.researchgate.net/publication/417008702_Timestral)

Two Strategies to Impute with Timestral

Because of its architecture, there are two primary methods for using Timestral to fill in your missing data gaps:

Strategy A: Iterative One-Step-Ahead Forecasting (Best for Long Internal Gaps)

If you have a historic sequence where data dropped out for 10 days, you can treat the day right before the gap as your forecast origin. You use Timestral to forecast the next 10 steps, taking the predicted median values to plug directly into the missing index.
python
import numpy as np
from timestral import TimestralForecaster

# Imagine a 10-step gap in the middle of a 100-step historical sequence
# Data is known from 0-49, missing from 50-59, known from 60-99
raw_data = np.random.randn(1, 1, 100)
raw_data[:, :, 50:60] = np.nan  # The gap

# 1. Slice context exactly up to the gap edge
clean_history = raw_data[:, :, :50] 

# 2. Forecast across the width of the gap
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")
imputation_slice = forecaster.forecast(clean_history, prediction_length=10)

# 3. Fill the gap with the generated median values
raw_data[:, :, 50:60] = imputation_slice.median
Use code with caution.

Strategy B: Multivariate Cross-Imputation (Best for Correlated Metrics)

If you run Timestral in its multivariate target configuration, it processes multiple metrics jointly. If Sensor_A goes offline and outputs NaN values, but Sensor_B and Sensor_C stay online and keep recording data, Timestral uses the active cross-correlations in the remaining sensors to probabilistically estimate what Sensor_A should be reading at that exact timestamp. [1] (https://www.instagram.com/reel/DfkMl0pAVBh/), [2] (https://github.com/mistralai/timestral/blob/main/README.md)

Trade-offs: Timestral vs. Traditional Imputation Tools

• The Downside: Traditional tools like linear interpolation or MICE (Multivariate Imputation by Chained Equations) look at both the past and the future relative to the gap to draw a smooth line through it. Because Timestral reads data causally (left-to-right), it won't inherently "look backward" from the post-gap data to correct its estimate unless you feed that post-gap data as an inverted sequence. [1] (https://www.nature.com/articles/s41467-026-77303-8), [2] (https://medium.com/@ning.jia/5-approaches-to-impute-missing-data-in-time-series-d6c75227819b), [3] (https://arxiv.org/html/2610.37585v1)
• The Benefit: Unlike static interpolation, which just draws a straight line and kills your dataset's natural variance, Timestral's 21-quantile output gives you a full probability distribution of what the missing data was. This allows you to generate a probabilistic multiple imputation matrix rather than just a single raw guess. [1] (https://www.tandfonline.com/doi/full/10.1080/00273171.2026.2710728), [2] (https://mistral.ai/news/timestral/), [3] (https://www.sciencedirect.com/science/article/pii/S0895435621000056), [4] (https://arxiv.org/html/2610.37585v1)
Are you trying to fix sporadic single-point drops (like a dropped network packet) or a sustained system blackout gap? I can give you a script to map out either approach.


## Example of predictive maintainance

Yes, predictive maintenance is one of Timestral's core intended industrial use cases. In fact, Mistral specifically highlighted industrial plant telemetry (such as monitoring a pump's vibration, temperature, and pressure) to showcase the model's strengths. [1] (https://emergent.sh/news/mistral-ai-launches-timestral-new-time-series), [2] (https://mistral.ai/news/timestral/)
The model's major value in predictive maintenance is its probabilistic quantile output (21 quantiles). Instead of guessing a single future sensor temperature, it provides a "worst-case risk band" (like the P90 or P95 quantile). This allows your monitoring systems to fire maintenance alerts only when the risk threshold of a failure crosses a safety limit, drastically reducing false alarms. [1] (https://github.com/mistralai/timestral/blob/main/README.md), [2] (https://www.edenai.co/post/time-series-foundation-models-explained-timestral-vs-timesfm-3-in-production), [3] (https://mistral.ai/news/timestral/)

How to Categorise and Feed Predictive Maintenance Data

To deploy a zero-shot forecast using , you map your machine telemetry into three main buckets: [1] (https://www.qovery.com/blog/self-host-timestral-time-series-forecasting-own-cloud), [2] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/)

1. Target Variables: The Core Telemetry (History Only)

These are the fluid physical measurements that reflect degradation or stress over time. Timestral operates as a multivariate forecaster, meaning it learns how these targets interact with each other in a single pass (e.g., how a spike in vibration causes a lagging spike in bearing temperature). [1] (https://mistral.ai/news/timestral/), [2] (https://www.qovery.com/blog/self-host-timestral-time-series-forecasting-own-cloud)
• Examples: Vibration (G-forces), bearing temperature (°C), acoustic emissions, oil pressure (PSI), or power draw (kW).
• Format: A 3D tensor of shape (Batch, Num_Sensors, History_Length). [1] (https://laurencemoroney.com/2026/10/16/timestral-time-series-foundation-models.html)

2. Past-Only Covariates: Ambient Conditions (History Only)

These are external factors that influence your machine's performance, but they are environment-driven, meaning you cannot control them or definitively look into their future.
• Examples: Ambient room humidity, surrounding factory floor temperature, or external grid voltage fluctuations.
• Format: Passed to the past_covariates argument. [1] (https://github.com/mistralai/timestral/blob/main/README.md)

3. Known-Future Covariates: Operational Schedules (Full Horizon)

This is where Timestral outperforms standard models. If a machine's temperature spikes simply because the factory operator intentionally doubled production throughput, a standard anomaly detector will freak out and flag an error. By feeding the operational plan as a known-future covariate, Timestral expects the temperature spike and adjusts the baseline. [1] (https://www.researchgate.net/publication/417008702_Timestral), [2] (https://mistral.ai/news/timestral/)
• Numeric Examples: Planned revolutions per minute (RPM), targeted output load percentage, or ambient weather forecasts if weather impacts the machinery.
• Categorical Examples: Machine operating state (0 for Idle, 1 for Running, 2 for Active Maintenance / Calibrating).
• Format: Continuous variables go to known_covariates, and integer-mapped statuses go to categorical_known_covariates. [1] (https://techjacksolutions.com/ai-tools/mistral/what-is-mistral-timestral/), [2] (https://mistral.ai/news/timestral/), [3] (https://github.com/mistralai/timestral/blob/main/README.md)

Practical Code Pipeline Example

Here is how you would organize a typical predictive maintenance data snapshot covering 24 hours of historical 15-minute sensor data (96 steps) to predict the next 4 hours (16 steps):
python
import numpy as np
from timestral import TimestralForecaster

# 1. Initialize Flagship Model (Runs easily on standard server CPUs)
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")

# Timeline Config
T_hist = 96   # 24 hours of history
H_pred = 16   # 4 hours prediction horizon
total_W = T_hist + H_pred  # 112 steps total for known features

# 2. Map Target Sensors: 1 Machine, 3 Interdependent Sensors (Temp, Vibration, Pressure)
# Shape: (Batch/Machines, Variates/Sensors, Time) -> (1, 3, 96)
sensor_history = np.random.randn(1, 3, T_hist)

# 3. Map Past-Only Covariates: Ambient Factory Humidity
# Shape: (1, 1, 96)
ambient_humidity = np.random.randn(1, 1, T_hist)

# 4. Map Known-Future Numeric Covariates: Scheduled RPM Speed
# Shape: (1, 1, 112) -> Must cover past + future horizon
production_rpm = np.random.randn(1, 1, total_W)

# 5. Map Known-Future Categorical Covariates: Operation State Code
# 0=Idle, 1=Standard Production, 2=High-Load Stress Test
# Shape: (1, 1, 112)
machine_state_codes = np.array([[[1]*96 + [2]*16]], dtype=float) # Stays standard, jumps to stress test in future

# 6. Execute Zero-Shot Forecast
predictions = forecaster.forecast(
    context=sensor_history,
    prediction_length=H_pred,
    past_covariates=ambient_humidity,
    known_covariates=production_rpm,
    categorical_known_covariates=machine_state_codes
)

# 7. Evaluate Upper Boundary Risk
# Extract the P95 (95th percentile) upper bound for Sensor 0 (Temperature)
p95_temp_risk = predictions.quantile(0.95)[0, 0, :]  # Shape: (16,)

# If p95_temp_risk crosses your hardware limit threshold -> Flag a Maintenance Ticket!
Use code with caution.

Next Steps for Production Architecture

As highlighted in industry implementations, Timestral shouldn't act as a standalone binary "Failure/No Failure" classifier. Instead, hook Timestral up to your sensor pipelines to output the probabilistic forecast boundaries. If Timestral indicates a 95% probability that your pump's bearing temperature will breach critical thresholds within the next 4 hours, pass that specific dataset to an automated script or a text-based LLM agent to generate a draft maintenance work order detailing exactly which sensor is failing. [1] (https://www.edenai.co/post/time-series-foundation-models-explained-timestral-vs-timesfm-3-in-production)
Do you have a specific asset or component you are tracking (e.g., a wind turbine gearbox, a server cooling pump, or conveyor motors)? If you tell me your expected sampling frequency (e.g., per-second vs. per-minute), I can recommend which Timestral size (5M vs 209M) best matches your latency limits.


In the case we want to use binary variables for axample a machine’s operational state (working vs. offline) or discrete breakdown history is excellent data to pass into Timestral.
However, how you feed this binary variable depends entirely on whether it represents a historical fact (an unexpected breakdown that already happened) or a future plan (a scheduled maintenance window).
You should map these states into Timestral’s input pipelines depending on their context:

Scenario A: The Binary State is a "Known-Future" Variable (Scheduled Offline Windows)

If your binary variable represents when a machine is intentionally scheduled to be Working (1) vs. Offline/Maintained (0), it should be treated as a Known-Future Categorical Covariate.
• Why it matters: If you take a machine offline for a scheduled service, its temperatures, power draw, and vibrations will suddenly crash to zero. A typical forecasting model will view this sudden plunge as a catastrophic system failure. By feeding this sequence ahead of time, Timestral understands that the drop is normal and expected, preventing your monitoring systems from firing false emergency alarms.
• How to feed it: Format it as an integer array (1.0 and 0.0) spanning your history length + your future forecast window, and pass it to categorical_known_covariates.

Scenario B: The Binary State is a "Past-Only" Variable (Unexpected Breakdowns)

If the binary variable tracks whether the machine was Working (1) or suffered an Unscheduled Breakdown (0) in the past, it is a Past-Only Categorical Covariate.
• Why it matters: A machine that has broken down three times in the last 48 hours is physically unstable. Even if it is running right now, its historical breakdown flips tell Timestral that the system is highly volatile, which will cause the model to automatically widen its probabilistic risk bands (quantiles) for the upcoming hours.
• How to feed it: Format it as an integer array spanning only your historical timeline, and pass it to categorical_past_covariates.

Python Code Example: Combining Both Binary Inputs

Here is how you inject both types of binary statuses simultaneously into your predictive maintenance script:
python
import numpy as np
from timestral import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")

T_hist = 96   # 24 hours of 15-minute historical data
H_pred = 16   # 4 hours of future predictions

# 1. Main Continuous Target: Machine Temperature
sensor_history = np.random.randn(1, 1, T_hist)

# =====================================================================
# INPUT 1: Past-Only Binary (Unscheduled Breakdowns that already occurred)
# =====================================================================
# 1.0 = Running smoothly, 0.0 = Had a sudden emergency trip/breakdown
past_breakdowns = np.ones((1, 1, T_hist)) 
past_breakdowns[:, :, 40:42] = 0.0  # Machine randomly tripped out for 30 mins yesterday
past_breakdowns = past_breakdowns.astype(float)

# =====================================================================
# INPUT 2: Known-Future Binary (Planned Operations vs Scheduled Maintenance)
# =====================================================================
# 1.0 = Scheduled to run production, 0.0 = Scheduled to turn off for service
# Must cover total timeline: 96 steps of history + 16 steps of future = 112 steps
future_schedule = np.ones((1, 1, T_hist + H_pred))
future_schedule[:, :, -4:] = 0.0     # The final hour of our future forecast is a planned shutdown
future_schedule = future_schedule.astype(float)

# =====================================================================
# 3. Fire the Forecast
# =====================================================================
predictions = forecaster.forecast(
    context=sensor_history,
    prediction_length=H_pred,
    categorical_past_covariates=past_breakdowns,
    categorical_known_covariates=future_schedule
)

# Extract your risk thresholds
p95_upper_risk = predictions.quantile(0.95)[0, 0, :]
Use code with caution.

Pro-Tip: Can Timestral predict the breakdown variable itself?

If your primary goal is to directly predict the probability of a binary breakdown happening next Tuesday, you can technically set the binary 0/1 array as your Target Variable instead of a covariate. Because Timestral outputs 21 continuous quantiles, a forecasted median value of 0.15 at a future hour can be interpreted as a 15% mathematical risk of the machine being offline/broken at that specific moment.
Are your breakdown events quite rare (e.g., happening once a month) or frequent (e.g., micro-stoppages several times a day)? If they are highly rare, I can show you how to tweak your data windowing strategy so the model actually sees enough breakdown examples in its history to learn from them.
