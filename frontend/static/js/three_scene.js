/**
 * AGRIVISION AI - Clean Premium 3D Agricultural Hero Scene
 * Optimized: Viewport-aware rendering loop, clamped pixel ratio, passive event handlers.
 */

(function initHero3D() {
  const canvas = document.getElementById('hero-3d-canvas');
  if (!canvas || typeof THREE === 'undefined') return;

  // 1. Scene, Camera & Depth Fog
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x080c10, 0.0016);

  const camera = new THREE.PerspectiveCamera(
    55,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 18, 55);
  camera.lookAt(0, 4, 0);

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
    precision: "mediump"
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

  // 2. Realistic Natural Agricultural Lighting
  const ambientLight = new THREE.AmbientLight(0x06281e, 1.8);
  scene.add(ambientLight);

  // Soft morning sunlight
  const sunLight = new THREE.DirectionalLight(0x6ee7b7, 2.2);
  sunLight.position.set(30, 45, 25);
  scene.add(sunLight);

  // Subtle cyan fill light
  const skyLight = new THREE.DirectionalLight(0x38bdf8, 1.0);
  skyLight.position.set(-30, 20, -10);
  scene.add(skyLight);

  // 3. Realistic Stylized 3D Plant Model (Stem + Leaves)
  const plantGroup = new THREE.Group();
  plantGroup.position.set(22, -6, 8);

  // Main Stem
  const stemCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.5, 6, 0.2),
    new THREE.Vector3(-0.8, 14, -0.4),
    new THREE.Vector3(0.3, 22, 0.5),
    new THREE.Vector3(0, 28, 0)
  ]);
  const stemGeo = new THREE.TubeGeometry(stemCurve, 24, 0.65, 8, false);
  const stemMat = new THREE.MeshStandardMaterial({
    color: 0x059669,
    roughness: 0.5,
    metalness: 0.1
  });
  const stemMesh = new THREE.Mesh(stemGeo, stemMat);
  plantGroup.add(stemMesh);

  // Organic Leaves Construction
  function createLeaf(scaleX, scaleY, scaleZ, rotX, rotY, rotZ, posX, posY, posZ) {
    const leafGeo = new THREE.SphereGeometry(1, 12, 12);
    leafGeo.scale(scaleX, scaleY, scaleZ);
    const leafMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x047857,
      emissiveIntensity: 0.18,
      roughness: 0.45,
      metalness: 0.15
    });
    const leaf = new THREE.Mesh(leafGeo, leafMat);
    leaf.rotation.set(rotX, rotY, rotZ);
    leaf.position.set(posX, posY, posZ);
    return leaf;
  }

  // Branching Leaf Pairs
  const leaves = [
    createLeaf(1.8, 0.2, 3.5, 0.4, 0.8, 0.3, 1.2, 7, 1.0),
    createLeaf(1.6, 0.2, 3.2, -0.3, -0.9, -0.4, -1.4, 11, -0.8),
    createLeaf(1.9, 0.2, 3.8, 0.5, 1.2, 0.2, 1.6, 16, 0.5),
    createLeaf(1.5, 0.2, 3.0, -0.4, -1.1, -0.3, -1.5, 20, -0.6),
    createLeaf(1.3, 0.18, 2.6, 0.3, 0.4, 0.2, 0.4, 25, 0.8),
    createLeaf(1.1, 0.15, 2.2, -0.2, -0.5, -0.2, -0.4, 27, -0.5)
  ];

  leaves.forEach(leaf => plantGroup.add(leaf));
  scene.add(plantGroup);

  // 4. Clean Agricultural Furrow Terrain
  const terrainWidth = 140;
  const terrainDepth = 140;
  const segments = 32;
  const terrainGeo = new THREE.PlaneGeometry(terrainWidth, terrainDepth, segments, segments);
  terrainGeo.rotateX(-Math.PI / 2);

  const posAttr = terrainGeo.attributes.position;
  const vertex = new THREE.Vector3();

  for (let i = 0; i < posAttr.count; i++) {
    vertex.fromBufferAttribute(posAttr, i);
    const furrow = Math.sin(vertex.x * 0.1) * 2.8 + Math.cos(vertex.z * 0.08) * 3.2;
    posAttr.setY(i, furrow);
  }
  terrainGeo.computeVertexNormals();

  const terrainMat = new THREE.MeshStandardMaterial({
    color: 0x071510,
    wireframe: true,
    transparent: true,
    opacity: 0.22,
    roughness: 0.85
  });

  const terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
  terrainMesh.position.y = -12;
  scene.add(terrainMesh);

  // 5. Subtle Floating Spores / Organic Bio-Particles
  const particleCount = 80;
  const particleGeo = new THREE.BufferGeometry();
  const particleCoords = [];

  for (let i = 0; i < particleCount; i++) {
    particleCoords.push(
      (Math.random() - 0.5) * 100,
      Math.random() * 35,
      (Math.random() - 0.5) * 80
    );
  }
  particleGeo.setAttribute('position', new THREE.Float32BufferAttribute(particleCoords, 3));

  const particleMat = new THREE.PointsMaterial({
    color: 0x6ee7b7,
    size: 0.45,
    transparent: true,
    opacity: 0.65,
    blending: THREE.AdditiveBlending
  });

  const particleCloud = new THREE.Points(particleGeo, particleMat);
  scene.add(particleCloud);

  // 6. Interactive Pointer Parallax
  let mouseX = 0;
  let mouseY = 0;
  let targetX = 0;
  let targetY = 0;
  let isVisible = true;
  let animId = null;

  window.addEventListener('mousemove', (e) => {
    if (!isVisible) return;
    mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
  }, { passive: true });

  // 7. Responsive Window Resize
  let resizeTimeout = null;
  window.addEventListener('resize', () => {
    if (resizeTimeout) clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }, 150);
  }, { passive: true });

  // 8. Viewport Intersection Observer for 0 CPU/GPU usage when scrolled out
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

  const heroSection = document.getElementById('hero') || canvas;
  observer.observe(heroSection);

  // 9. Smooth Animation Loop
  let clock = new THREE.Clock();

  function animate() {
    if (!isVisible) {
      animId = null;
      return;
    }
    animId = requestAnimationFrame(animate);
    const elapsedTime = clock.getElapsedTime();

    // Gentle plant organic sway in breeze
    plantGroup.rotation.z = Math.sin(elapsedTime * 1.2) * 0.04;
    plantGroup.rotation.y = Math.cos(elapsedTime * 0.8) * 0.05;

    // Organic spore particle drift
    particleCloud.rotation.y = elapsedTime * 0.02;

    // Gentle terrain drift
    terrainMesh.rotation.y = elapsedTime * 0.01;

    // Smooth camera mouse parallax
    targetX += (mouseX * 6 - targetX) * 0.035;
    targetY += (-mouseY * 3.5 - targetY) * 0.035;
    camera.position.x = targetX;
    camera.position.y = 18 + targetY;
    camera.lookAt(0, 4, 0);

    renderer.render(scene, camera);
  }

  animate();
})();
