let macroChart;

// Sidebar toggle logic
document.getElementById("nav-calorie").addEventListener("click", () => {
    document.getElementById("calorie-section").classList.remove("hidden");
    document.getElementById("meal-section").classList.add("hidden");
    document.getElementById("nav-calorie").classList.add("active");
    document.getElementById("nav-meal").classList.remove("active");
});

document.getElementById("nav-meal").addEventListener("click", () => {
    document.getElementById("meal-section").classList.remove("hidden");
    document.getElementById("calorie-section").classList.add("hidden");
    document.getElementById("nav-meal").classList.add("active");
    document.getElementById("nav-calorie").classList.remove("active");
});

// ----------------------
// Calorie Predictor Logic
// ----------------------
document.getElementById("predictBtn").addEventListener("click", async () => {
    const payload = {
        Age: document.getElementById("Age").value,
        Weight: document.getElementById("Weight").value,
        Height: document.getElementById("Height").value,
        ActivityFactor: document.getElementById("ActivityFactor").value,
        Preference: document.getElementById("Preference").value,
        Goal: document.getElementById("Goal").value,
        BMR: document.getElementById("BMR").value,
        city: document.getElementById("city").value,
        Temperature: document.getElementById("Temperature").value,
        Humidity: document.getElementById("Humidity").value,
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    const cal = data.PredictedCalories;

    document.getElementById("calValue").textContent = cal.toFixed(2);

    const macros = {
        protein: payload.Weight * 1.6,
        fats: cal * 0.25 / 9,
        carbs: (cal - ((payload.Weight * 1.6 * 4) + (cal * 0.25))) / 4,
        fiber: cal / 100
    };

    document.getElementById("proteinValue").textContent = macros.protein.toFixed(2);
    document.getElementById("fatValue").textContent = macros.fats.toFixed(2);
    document.getElementById("carbValue").textContent = macros.carbs.toFixed(2);
    document.getElementById("fiberValue").textContent = macros.fiber.toFixed(2);

    document.getElementById("results").classList.remove("hidden");

    if (macroChart) macroChart.destroy();
    const ctx = document.getElementById("macroChart").getContext("2d");
    macroChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Protein", "Fats", "Carbs", "Fiber"],
            datasets: [{
                label: "Macros (g)",
                data: [macros.protein, macros.fats, macros.carbs, macros.fiber],
                backgroundColor: ["#00f7ff", "#ff6b6b", "#ffd93d", "#6bff95"]
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });
});

// ----------------------
// Meal Recommender Logic
// ----------------------
document.getElementById("mrBtn").addEventListener("click", async () => {
    const payload = {
        calorie_target: document.getElementById("mr_calories").value,
        preference: document.getElementById("mr_pref").value,
        goal: document.getElementById("mr_goal").value,
        city: document.getElementById("mr_city").value,
        temp: document.getElementById("mr_temp").value
    };

    const res = await fetch("/meal_recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    document.getElementById("mr_results").classList.remove("hidden");

    const container = document.getElementById("mealPlanCards");
    container.innerHTML = "";

    for (const [meal, items] of Object.entries(data.plan)) {
        const mealCard = document.createElement("div");
        mealCard.className = "card";
        mealCard.innerHTML = `<h3>${meal.toUpperCase()}</h3>`;
        items.forEach(item => {
            mealCard.innerHTML += `
                <p><strong>${item.Description}</strong> - ${item.grams}g 
                (${item.calories} kcal, P:${item.protein_g}g F:${item.fat_g}g C:${item.carbs_g}g)</p>`;
        });
        container.appendChild(mealCard);
    }
});
