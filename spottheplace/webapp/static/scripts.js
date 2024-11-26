const uploadForm = document.getElementById("uploadForm");
const mainContent = document.querySelector(".main-content");
const animationContainer = document.getElementById("animationContainer");
const resultContainer = document.getElementById("resultContainer");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");

const flags = document.querySelectorAll(".flag");

let interval;

uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  mainContent.style.display = "none";
  animationContainer.style.display = "flex";

  startFlagAnimation();

  setTimeout(() => {
    stopFlagAnimation();
    displayResults("Japan", "/static/flags/japan.png");
  }, 5000);
});

function startFlagAnimation() {
  let activeIndex = -1;

  interval = setInterval(() => {
    if (activeIndex !== -1) {
      flags[activeIndex].classList.remove("active");
    }
    activeIndex = Math.floor(Math.random() * flags.length);
    flags[activeIndex].classList.add("active");
  }, 250);
}

function stopFlagAnimation() {
  clearInterval(interval);
  flags.forEach((flag) => flag.classList.remove("active"));
}

function displayResults(countryName, flagSrc) {
  animationContainer.style.display = "none";
  resultContainer.style.display = "flex";

  document.getElementById("finalImagePreview").src =
    document.getElementById("imagePreview").src;
  document.getElementById("countryName").textContent = countryName;
  document.getElementById("finalFlag").src = flagSrc;
}

function resetPage() {
  resultContainer.style.display = "none";
  mainContent.style.display = "flex";
}

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";
    };
    reader.readAsDataURL(file);
  } else {
    imagePreview.src = "";
    imagePreview.style.display = "none";
  }
});

function resetPage() {
  document.getElementById("resultContainer").style.display = "none";

  document.getElementById("mainContent").style.display = "flex";

  const imageInput = document.getElementById("imageInput");
  const imagePreview = document.getElementById("imagePreview");

  imageInput.value = "";
  imagePreview.src = "";
  imagePreview.style.display = "none";
}

function displayResults(countryName, flagSrc) {
  // Afficher les résultats
  document.getElementById("animationContainer").style.display = "none";
  document.getElementById("resultContainer").style.display = "flex";

  document.getElementById("finalImagePreview").src =
    document.getElementById("imagePreview").src;
  document.getElementById("countryName").textContent = countryName;
  document.getElementById("finalFlag").src = flagSrc;

  launchFireworks();
}

function launchFireworks() {
  const duration = 1 * 1000;
  const end = Date.now() + duration;

  const colors = [
    "#ff0000",
    "#ff6600",
    "#ffd700",
    "#00ff00",
    "#00aaff",
    "#ff00ff",
  ];

  (function frame() {
    confetti({
      particleCount: 5,
      angle: Math.random() * 360,
      spread: 55,
      origin: {
        x: Math.random(),
        y: Math.random() * 0.5,
      },
      colors: colors,
    });

    if (Date.now() < end) {
      requestAnimationFrame(frame);
    }
  })();
}
