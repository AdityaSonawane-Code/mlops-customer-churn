const API_URL = "http://127.0.0.1:8000";


async function loadMetrics() {

    try {

        const response = await fetch(`${API_URL}/metrics`);

        if (!response.ok) {
            throw new Error("API request failed");
        }

        const data = await response.json();


        // Summary cards

        document.getElementById("totalPredictions").textContent =
            data.total_predictions;

        document.getElementById("predictedChurn").textContent =
            data.predicted_churn;

        document.getElementById("highRisk").textContent =
            data.high_risk;

        document.getElementById("averageProbability").textContent =
            (data.average_churn_probability * 100).toFixed(2) + "%";


        // Risk percentages

        const total = data.total_predictions;

        if (total > 0) {

            const highPercent =
                (data.high_risk / total) * 100;

            const mediumPercent =
                (data.medium_risk / total) * 100;

            const lowPercent =
                (data.low_risk / total) * 100;


            document.getElementById("highRiskPercent").textContent =
                highPercent.toFixed(1) + "%";

            document.getElementById("mediumRiskPercent").textContent =
                mediumPercent.toFixed(1) + "%";

            document.getElementById("lowRiskPercent").textContent =
                lowPercent.toFixed(1) + "%";


            document.getElementById("highBar").style.width =
                highPercent + "%";

            document.getElementById("mediumBar").style.width =
                mediumPercent + "%";

            document.getElementById("lowBar").style.width =
                lowPercent + "%";

        } else {

            document.getElementById("highRiskPercent").textContent = "0%";
            document.getElementById("mediumRiskPercent").textContent = "0%";
            document.getElementById("lowRiskPercent").textContent = "0%";

            document.getElementById("highBar").style.width = "0%";
            document.getElementById("mediumBar").style.width = "0%";
            document.getElementById("lowBar").style.width = "0%";
        }


    } catch (error) {

        console.error("Dashboard error:", error);

        document.getElementById("totalPredictions").textContent = "Offline";
        document.getElementById("predictedChurn").textContent = "--";
        document.getElementById("highRisk").textContent = "--";
        document.getElementById("averageProbability").textContent = "--";

    }

}


// Load metrics when dashboard opens

loadMetrics();


// Automatically refresh every 5 seconds

setInterval(loadMetrics, 5000);