/**
 * AGRIVISION AI - Interactive 3D Smart Farm Canvas
 * Optimized: Viewport-aware rendering, event-driven raycasting, clamped pixel ratio.
 */

(function initSmartFarm3D() {
  const canvas = document.getElementById('smart-farm-canvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const container = canvas.parentElement;
  let width = container.clientWidth || window.innerWidth;
  let height = container.clientHeight || 520;

  // Scene & Camera
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07131c);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 500);
  camera.position.set(0, 32, 45);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
    precision: "mediump"
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

  // Lighting
  const ambient = new THREE.AmbientLight(0x1e3a47, 2.0);
  scene.add(ambient);

  const sunLight = new THREE.DirectionalLight(0xfef3c7, 2.5);
  sunLight.position.set(30, 45, 20);
  scene.add(sunLight);

  const farmLightEmerald = new THREE.PointLight(0x10b981, 2.0, 50);
  farmLightEmerald.position.set(-15, 10, -10);
  scene.add(farmLightEmerald);

  // Farm Ground Base Platform (Hexagonal Platform)
  const baseGeo = new THREE.CylinderGeometry(24, 26, 2, 6);
  const baseMat = new THREE.MeshStandardMaterial({
    color: 0x0c212d,
    roughness: 0.7,
    metalness: 0.3
  });
  const farmBase = new THREE.Mesh(baseGeo, baseMat);
  farmBase.position.y = -1;
  scene.add(farmBase);

  // Grid outline
  const gridHelper = new THREE.GridHelper(40, 20, 0x10b981, 0x153542);
  gridHelper.position.y = 0.05;
  scene.add(gridHelper);

  // Interactive Hub Objects List for Raycasting
  const interactiveHubs = [];

  function createFarmHub(name, icon, color, x, z, targetSectionId) {
    const group = new THREE.Group();
    group.position.set(x, 1.2, z);

    // Pedestal
    const pedGeo = new THREE.CylinderGeometry(2.5, 3, 1.2, 12);
    const pedMat = new THREE.MeshStandardMaterial({
      color: 0x0e2836,
      roughness: 0.4,
      metalness: 0.6
    });
    const ped = new THREE.Mesh(pedGeo, pedMat);
    group.add(ped);

    // Holographic Core Ring
    const ringGeo = new THREE.TorusGeometry(3.2, 0.08, 12, 24);
    const ringMat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.6;
    group.add(ring);

    // Floating Interactive Diamond
    const crystalGeo = new THREE.OctahedronGeometry(1.6, 0);
    const crystalMat = new THREE.MeshStandardMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: 0.4,
      roughness: 0.2,
      metalness: 0.8
    });
    const crystal = new THREE.Mesh(crystalGeo, crystalMat);
    crystal.position.y = 3.5;
    group.add(crystal);

    group.userData = {
      name: name,
      icon: icon,
      targetSectionId: targetSectionId,
      crystal: crystal,
      ring: ring,
      baseColor: color
    };

    scene.add(group);
    interactiveHubs.push(group);
    return group;
  }

  // 1. Crop Intelligence Field
  createFarmHub("Crop Intelligence Hub", "🌱", 0x10b981, -12, -8, "module-crop");

  // 2. Soil & Fertilizer Depot
  createFarmHub("Fertilizer & Soil Diagnostics", "🧪", 0x06b6d4, 12, -8, "module-fertilizer");

  // 3. Plant Doctor Lab
  createFarmHub("Plant Doctor (Disease AI)", "🔬", 0xf59e0b, -14, 8, "module-disease");

  // 4. Pest Scouting Zone
  createFarmHub("Pest AI & IPM Sentinel", "🐛", 0xf43f5e, 14, 8, "module-pest");

  // 5. Agronomy RAG Chat Core
  createFarmHub("AGRIVISION RAG Assistant", "💬", 0x8b5cf6, 0, 0, "module-chat");

  // Raycasting & Pointer Interaction (Event-driven, NOT on every render frame)
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2(-999, -999);
  let hoveredHub = null;

  function performRaycast() {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    let newHover = null;
    for (const hit of intersects) {
      let current = hit.object;
      while (current.parent && current.parent !== scene) {
        current = current.parent;
      }
      if (interactiveHubs.includes(current)) {
        newHover = current;
        break;
      }
    }

    if (newHover !== hoveredHub) {
      if (hoveredHub) {
        hoveredHub.userData.crystal.scale.set(1, 1, 1);
        canvas.style.cursor = 'default';
      }
      hoveredHub = newHover;
      if (hoveredHub) {
        hoveredHub.userData.crystal.scale.set(1.3, 1.3, 1.3);
        canvas.style.cursor = 'pointer';
      }
    }
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    performRaycast();
  }, { passive: true });

  canvas.addEventListener('mouseleave', () => {
    mouse.x = -999;
    mouse.y = -999;
    if (hoveredHub) {
      hoveredHub.userData.crystal.scale.set(1, 1, 1);
      canvas.style.cursor = 'default';
      hoveredHub = null;
    }
  }, { passive: true });

  canvas.addEventListener('click', () => {
    performRaycast();
    if (hoveredHub) {
      const targetId = hoveredHub.userData.targetSectionId;
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth' });
        targetEl.style.transition = 'box-shadow 0.4s ease';
        targetEl.style.boxShadow = '0 0 40px rgba(16, 185, 129, 0.6)';
        setTimeout(() => { targetEl.style.boxShadow = ''; }, 2000);
      }
    }
  });

  // Hotspot overlay button listeners
  document.querySelectorAll('.farm-hotspot-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.farm-hotspot-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.getAttribute('data-target');
      const el = document.getElementById(target);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // Resize handler
  let resizeTimeout = null;
  window.addEventListener('resize', () => {
    if (resizeTimeout) clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      width = container.clientWidth;
      height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }, 150);
  }, { passive: true });

  // Viewport Intersection Observer
  let isVisible = false;
  let animId = null;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      isVisible = entry.isIntersecting;
      if (isVisible && !animId) {
        clock.start();
        animate();
      } else if (!isVisible && animId) {
        cancelAnimationFrame(animId);
        animId = null;
        clock.stop();
      }
    });
  }, { threshold: 0.05 });

  const smartFarmSection = document.getElementById('smart-farm') || canvas;
  observer.observe(smartFarmSection);

  // Animation Loop
  let clock = new THREE.Clock();

  function animate() {
    if (!isVisible) {
      animId = null;
      return;
    }
    animId = requestAnimationFrame(animate);
    const elapsedTime = clock.getElapsedTime();

    // Slow orbit rotation of the farm base
    farmBase.rotation.y = elapsedTime * 0.05;
    gridHelper.rotation.y = elapsedTime * 0.05;

    // Animate Hub Crystals (smooth mathematical floating)
    interactiveHubs.forEach((hub, idx) => {
      const crystal = hub.userData.crystal;
      crystal.rotation.y = elapsedTime * 1.2 + idx;
      crystal.rotation.x = Math.sin(elapsedTime * 0.8 + idx) * 0.3;
      crystal.position.y = 3.2 + Math.sin(elapsedTime * 1.5 + idx) * 0.4;
    });

    renderer.render(scene, camera);
  }
})();
