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
  }, 200);
}

function stopFlagAnimation() {
  clearInterval(interval);
  flags.forEach((flag) => flag.classList.remove("active"));
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

export function resetPage() {
  document.getElementById("resultContainer").style.display = "none";
  document.getElementById("mainContent").style.display = "flex";

  const imageInput = document.getElementById("imageInput");
  const imagePreview = document.getElementById("imagePreview");

  imageInput.value = "";
  imagePreview.src = "";
  imagePreview.style.display = "none";
}

window.resetPage = resetPage;

document.getElementById("resetButton").addEventListener("click", resetPage);

function displayResults(countryName, flagSrc) {
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
      particleCount: 20,
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

const globeContainer = document.getElementById("globeContainer");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  globeContainer.offsetWidth / globeContainer.offsetHeight,
  0.1,
  1000
);
camera.position.z = 250;

const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(globeContainer.offsetWidth, globeContainer.offsetHeight);
globeContainer.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(2, 1, 1).normalize();
scene.add(directionalLight);

const pointLight = new THREE.PointLight(0xffffff, 0.8);
pointLight.position.set(200, 200, 200);
scene.add(pointLight);

// Texture loader
const textureLoader = new THREE.TextureLoader();
const globeTexture = textureLoader.load("./static/shokunin_World_Map.png");
const displacementMap = textureLoader.load(
  "./static/shokunin_World_Map_bw.png"
);

// Geometry and material for the globe
const globeGeometry = new THREE.SphereGeometry(100, 128, 128);
const globeMaterial = new THREE.MeshStandardMaterial({
  map: globeTexture,
  displacementMap: displacementMap,
  displacementScale: 3,
  roughness: 0.8,
  metalness: 0.1,
});

// Create globe mesh
const globeMesh = new THREE.Mesh(globeGeometry, globeMaterial);
globeMesh.scale.set(1.1, 1.1, 1.1);
scene.add(globeMesh);

// Handle responsive resizing
window.addEventListener("resize", () => {
  camera.aspect = globeContainer.offsetWidth / globeContainer.offsetHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(globeContainer.offsetWidth, globeContainer.offsetHeight);
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  globeMesh.rotation.y += 0.015;
  renderer.render(scene, camera);
}

animate();
