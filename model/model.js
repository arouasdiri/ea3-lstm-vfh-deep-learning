Math.seedrandom('ea3-fixed-seed');

let model;
let wordIndex = {};
let indexWord = {};
const maxInputLength = 9;
let autoInterval = null;
let autoSteps = 0;
const maxAutoSteps = 10;

async function loadModelAndTokenizer() {
    try {
        model = await tf.loadLayersModel("lm_tfjs-data/model.json");

        wordIndex = await fetch("model/tokenizer_word2index.json").then(res => res.json());
        const i2w = await fetch("model/tokenizer_index2word.json").then(res => res.json());
        for (const [k, v] of Object.entries(i2w)) {
            indexWord[+k] = v;
        }

        const rawText = await fetch("data/train.txt").then(res => res.text());
        const words = rawText
            .toLowerCase()
            .replace(/[^a-zäöüß ]/g, " ")
            .replace(/\s+/g, " ")
            .trim()
            .split(" ");

        const sequences = [];
        const labels = [];
        for (let i = maxInputLength; i < words.length; i++) {
            const seq = words.slice(i - maxInputLength, i).map(w => wordIndex[w] || 0);
            const label = wordIndex[words[i]] || 0;
            if (seq.includes(0) || label === 0) continue;
            sequences.push(seq);
            labels.push(label);
            if (sequences.length >= 2000) break;
        }

        const xs = tf.tensor2d(sequences);
        const ys = tf.oneHot(tf.tensor1d(labels, "int32"), Object.keys(wordIndex).length);
        evaluateModel(model, xs, ys, Object.keys(wordIndex).length);

    } catch (err) {
        console.error("Error loading model:", err);
        const evalBox = document.getElementById("evaluationResult");
        if (evalBox) evalBox.innerText = "Error loading or evaluating model:\n" + err.message;
    }
}

function textToSequence(text) {
    const tokens = text.toLowerCase().replace(/[^a-zäöüß ]/g, "").trim().split(/\s+/);
    const sequence = tokens.map(w => wordIndex[w] || 0).filter(id => id > 0);
    if (sequence.length === 0) showError("⚠️ No known words in the input.");
    return sequence;
}

function padSequence(seq) {
    const padded = new Array(maxInputLength - seq.length).fill(0).concat(seq.slice(-maxInputLength));
    return tf.tensor2d([padded]);
}

function showError(msg) {
    let box = document.getElementById("errbox");
    if (!box) {
        box = document.createElement("div");
        box.id = "errbox";
        box.className = "error-message";
        document.getElementById("output").prepend(box);
    }
    box.textContent = msg;
    box.style.display = "block";
    setTimeout(() => box.style.display = "none", 4000);
}

async function showPrediction() {
    const input = document.getElementById("input").value.trim();
    if (!input) return;
    const seq = textToSequence(input);
    if (seq.length === 0) return;
    const pred = await model.predict(padSequence(seq)).data();

    const top = Array.from(pred)
        .map((p, i) => ({ word: indexWord[i], prob: p }))
        .filter(x => x.word)
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);

    const output = document.getElementById("output");
    output.innerHTML = "<b>Next word predictions:</b><br>";
    top.forEach(({ word, prob }) => {
        output.innerHTML += `<button onclick="addWord('${word}')">${word} (${(prob * 100).toFixed(2)}%)</button> `;
    });

    Plotly.newPlot("plotContainer", [{
        x: top.map(item => item.word),
        y: top.map(item => item.prob),
        type: "bar",
        text: top.map(item => (item.prob * 100).toFixed(2) + "%"),
        textposition: "auto"
    }], {
        title: "Top-5 Word Probabilities",
        xaxis: { title: "Word" },
        yaxis: { title: "Probability", tickformat: ".0%" }
    });
}

function addWord(word) {
    const input = document.getElementById("input");
    input.value = (input.value + " " + word).trim();
    showPrediction();
}

async function continueWithBestWord() {
    const input = document.getElementById("input").value.trim();
    if (!input) return;
    const seq = textToSequence(input);
    if (seq.length === 0) return;
    const pred = await model.predict(padSequence(seq)).data();
    const bestWord = indexWord[pred.indexOf(Math.max(...pred))];
    if (bestWord) addWord(bestWord);
}

async function startAutoGeneration() {
    if (autoInterval) return;
    autoSteps = 0;
    autoInterval = setInterval(async () => {
        if (autoSteps++ >= maxAutoSteps) return stopAutoGeneration();
        await continueWithBestWord();
    }, 1000);
}

function stopAutoGeneration() {
    clearInterval(autoInterval);
    autoInterval = null;
}

function resetInput() {
    stopAutoGeneration();
    document.getElementById("input").value = "";
    document.getElementById("output").innerHTML = "";
}

async function evaluateModel(model, xs, ys, vocabSize) {
    console.log("📊 Starting evaluation...");

    const predictions = await model.predict(xs);
    const predProbs = await predictions.array();
    const trueIndices = await ys.argMax(-1).array();

    const topKValues = [1, 5, 10, 20, 100];
    const topKCorrect = Object.fromEntries(topKValues.map(k => [k, 0]));
    let totalLoss = 0;

    for (let i = 0; i < predProbs.length; i++) {
        const probs = predProbs[i];
        const trueIdx = trueIndices[i];

        const sorted = probs
            .map((p, idx) => ({ idx, p }))
            .sort((a, b) => b.p - a.p)
            .map(x => x.idx);

        topKValues.forEach(k => {
            if (sorted.slice(0, k).includes(trueIdx)) topKCorrect[k]++;
        });

        const pTrue = Math.max(probs[trueIdx], 1e-10);
        totalLoss += -Math.log(pTrue);
    }

    const total = predProbs.length;
    const perplexity = Math.exp(totalLoss / total);

    const evalDiv = document.getElementById("evaluationResult");
    evalDiv.innerHTML = `
        <b>📈 Evaluation Results:</b><br>
        Top-1 Accuracy: ${(topKCorrect[1] / total * 100).toFixed(2)}%<br>
        Top-5 Accuracy: ${(topKCorrect[5] / total * 100).toFixed(2)}%<br>
        Top-10 Accuracy: ${(topKCorrect[10] / total * 100).toFixed(2)}%<br>
        Top-20 Accuracy: ${(topKCorrect[20] / total * 100).toFixed(2)}%<br>
        Top-100 Accuracy: ${(topKCorrect[100] / total * 100).toFixed(2)}%<br>
        Perplexity: ${perplexity.toFixed(2)}
    `;
    console.log("✅ Evaluation complete.");
}

loadModelAndTokenizer();
