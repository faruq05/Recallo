// RecalloVisual3D.jsx
import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass";

const RecalloVisual3D = () => {
  const containerRef = useRef();
  const composerRef = useRef();
  const rendererRef = useRef();
  const meshRef = useRef();
  const uniformsRef = useRef({
    u_time: { value: 0.0 },
  });

  useEffect(() => {
    const container = containerRef.current;
    const width = 120;
    const height = 120;

    // Scene & Camera setup (same as before)
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, 6);

    // Renderer setup (same)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;;
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.backgroundColor = "transparent";
    renderer.domElement.style.display = "block";

    rendererRef.current = renderer;

    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    container.style.background = "transparent";
    container.style.margin = "0";
    container.style.padding = "0";
    container.style.overflow = "hidden";
    container.style.width = `${width}px`;
    container.style.height = `${height}px`;
    container.style.pointerEvents = "auto";

    container.appendChild(renderer.domElement);

    // Shader material and geometry (same)
    const material = new THREE.ShaderMaterial({
      wireframe: true,
      transparent: true,
      uniforms: uniformsRef.current,
      vertexShader,
      fragmentShader,
    });

    const geometry = new THREE.IcosahedronGeometry(1.8, 2);
    const mesh = new THREE.Mesh(geometry, material);
    meshRef.current = mesh;
    scene.add(mesh);

    // Postprocessing setup (same)
    const composer = new EffectComposer(renderer);
    composerRef.current = composer;
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(new THREE.Vector2(width, height));
    bloomPass.threshold = 0.95;
    bloomPass.strength = 0.05;
    bloomPass.radius = 0.05;
    composer.addPass(bloomPass);
    composer.addPass(new OutputPass());

    const clock = new THREE.Clock();

    // Target rotations
    let targetRotationX = 0;
    let targetRotationY = 0;

    // Mouse move handler on window, relative to viewport size
    function onMouseMove(event) {
      const vw = window.innerWidth;
      const vh = window.innerHeight;

      // Normalized coordinates: -1 to 1
      const x = (event.clientX / vw) * 2 - 1;
      const y = -((event.clientY / vh) * 2 - 1);

      targetRotationY = x * Math.PI * 0.3;
      targetRotationX = y * Math.PI * 0.3;
    }

    window.addEventListener("mousemove", onMouseMove);

    function animate() {
      uniformsRef.current.u_time.value = clock.getElapsedTime();

      if (meshRef.current) {
        meshRef.current.rotation.x +=
          (targetRotationX - meshRef.current.rotation.x) * 0.05;
        meshRef.current.rotation.y +=
          (targetRotationY - meshRef.current.rotation.y) * 0.05;
      }

      composer.render();
      requestAnimationFrame(animate);
    }
    animate();

    // Cleanup on unmount
    return () => {
      renderer.dispose();
      window.removeEventListener("mousemove", onMouseMove);
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: "120px",
        height: "120px",
        overflow: "hidden",
        padding: 0,
        margin: 0,
        background: "transparent",
      }}
    />
  );
};

const vertexShader = `
  uniform float u_time;

  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  vec3 fade(vec3 t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }

  float pnoise(vec3 P, vec3 rep) {
    vec3 Pi0 = mod(floor(P), rep);
    vec3 Pi1 = mod(Pi0 + vec3(1.0), rep);
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);

    vec4 gx0 = ixy0 * (1.0 / 7.0);
    vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);

    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);

    vec4 norm0 = taylorInvSqrt(vec4(dot(g000,g000), dot(g010,g010), dot(g100,g100), dot(g110,g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));

    vec3 fade_xyz = fade(Pf0);
    vec2 n_yz = mix(vec2(n000, n100), vec2(n010, n110), fade_xyz.yz);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
  }

  void main() {
    float noise = 1.5 * pnoise(position + u_time, vec3(10.0));
    float displacement = noise / 12.0;
    vec3 newPosition = position + normal * displacement;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

const fragmentShader = `
  uniform float u_time;

  const vec3 white = vec3(1.0, 1.0, 1.0);
  const vec3 yellow = vec3(0.929, 0.706, 0.216);

  void main() {
    float t = (sin(u_time) + 1.0) / 2.0;
    vec3 color = mix(white, yellow, t);
    gl_FragColor = vec4(color, 0.4);
  }
`;

export default RecalloVisual3D;
