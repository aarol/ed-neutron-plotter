import { OrbitControls } from 'three/examples/jsm/Addons.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { color, float, uniform, vec4 } from 'three/tsl';
import { AdditiveBlending, BufferAttribute, BufferGeometry, CubeTextureLoader, Mesh, PerspectiveCamera, Points, PointsNodeMaterial, Raycaster, Scene, SphereGeometry, SpriteNodeMaterial, Vector2, Vector3, WebGPURenderer } from 'three/webgpu';
import { LinePoints } from './line-points';
import type { StarSystem } from './ui/types';
import type { Module } from '../rust-module/pkg/rust_module';


export class Galaxy {
  private camera = new PerspectiveCamera(25, window.innerWidth / window.innerHeight, 0.1, 200)
  private scene = new Scene();
  // Draws the route lines on top of the stars
  // because the stars are drawn with additive blending
  private overlayScene = new Scene();
  private renderer = new WebGPURenderer({
    antialias: true,
    depth: false,
  })

  private wasmModule?: Module;
  onSystemFocus?: (system: StarSystem) => void;

  private controls = new OrbitControls(this.camera, this.renderer.domElement)
  targetPosition = new Vector3(0, 0, 0)
  currentPosition = this.targetPosition.clone()
  private routeLine = new LinePoints(256, 0xffffff, 0.003);

  private raycaster = new Raycaster();
  private pointerDownPos = new Vector2();
  private isDragging = false;

  focusSphere!: Mesh

  stats = new Stats()

  async init(primaryModule: Module) {
    this.wasmModule = primaryModule;
    this.camera.position.set(34.65699659876029, 21.90527423256544, -24.079356892645272);

    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setClearColor('#000000');
    this.renderer.autoClear = false;
    document.body.appendChild(this.renderer.domElement);

    document.body.appendChild(this.stats.dom)

    await this.renderer.init();

    this.controls.enableDamping = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 150;
    this.controls.zoomSpeed = 3.0
    this.controls.update()

    // this is for you javascript (middle finger)
    const localThis = this
    this.controls.addEventListener('change', this.requestRenderIfNotRequested.bind(localThis))

    window.addEventListener('resize', this.onWindowResize.bind(localThis));

    const cubeTextureLoader = new CubeTextureLoader();
    const base = import.meta.env.BASE_URL;
    const texture = await cubeTextureLoader.loadAsync([
      `${base}skybox/front.png`,
      `${base}skybox/back.png`,
      `${base}skybox/top.png`,
      `${base}skybox/bottom.png`,
      `${base}skybox/left.png`,
      `${base}skybox/right.png`,
    ])

    this.scene.background = texture;

    this.focusSphere = this.createFocusSphere()
    this.overlayScene.add(this.focusSphere)
    this.overlayScene.add(this.routeLine)

    this.initializeMouseEvents();
  }

  createFocusSphere() {
    const geometry = new SphereGeometry(0.001, 16, 16);
    const material = new SpriteNodeMaterial({
      colorNode: vec4(uniform(color('#ffffff')), float(1.0)),
      sizeAttenuation: false,
    })
    const sphere = new Mesh(geometry, material)
    sphere.position.copy(this.targetPosition)
    return sphere
  }

  setTarget(target: Vector3) {
    this.targetPosition = target
    this.focusSphere.position.copy(this.targetPosition)
    this.requestRenderIfNotRequested()
  }

  setRoutePoints(points: Float32Array) {
    this.routeLine.update(points);
    this.requestRenderIfNotRequested()
  }
  
  setRoutePointsFromCoords(coords: StarSystem["coords"][]) {
    // Convert nodes to Float32Array format
    const points = new Float32Array(coords.length * 3);
    coords.forEach((coord, i) => {
      points[i * 3] = coord.x;
      points[i * 3 + 1] = coord.y;
      points[i * 3 + 2] = coord.z;
    });
    this.routeLine.update(points);
    this.requestRenderIfNotRequested()
  }

  setRouteProgress(index: number) {
    this.routeLine.setProgress(index);
    this.requestRenderIfNotRequested();
  }

  private initializeMouseEvents() {
    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
    this.renderer.domElement.addEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.addEventListener('pointerup', this.onPointerUp);
  }

  clearRoute() {
    this.routeLine.update(new Float32Array([]));
    this.routeLine.setProgress(0);
    this.requestRenderIfNotRequested();
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.requestRenderIfNotRequested()
  }

  renderRequested = false
  render() {
    this.renderRequested = false

    this.controls.update()
    this.stats.update()
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera)
    this.renderer.clearDepth();
    this.renderer.render(this.overlayScene, this.camera)

    if (this.currentPosition.distanceToSquared(this.targetPosition) > 0.0001) {
      this.currentPosition.lerp(this.targetPosition, 0.08)
      this.controls.target.copy(this.currentPosition)
      this.requestRenderIfNotRequested()
    }
  }

  requestRenderIfNotRequested() {
    if (!this.renderRequested) {
      this.renderRequested = true

      requestAnimationFrame(this.render.bind(this));
    }
  }

  async loadStars(starPositionArrays: DataView[]) {
    console.log('Loading star data...')


    const count = starPositionArrays.reduce((acc, arr) => acc + arr.byteLength / 12 - 1, 0)
    console.log(`Loaded ${count} stars`)

    for (let i = 0; i < starPositionArrays.length; i++) {
      const arr = starPositionArrays[i];

      // let aabb_min_x = arr.getFloat32(0, true)
      // let aabb_min_y = arr.getFloat32(4, true)
      // let aabb_min_z = arr.getFloat32(8, true)
      // let aabb_min = new Vector3(aabb_min_x, aabb_min_y, aabb_min_z).divideScalar(1000)
      // let aabb_max_x = arr.getFloat32(12, true)
      // let aabb_max_y = arr.getFloat32(16, true)
      // let aabb_max_z = arr.getFloat32(20, true)
      // let aabb_max = new Vector3(aabb_max_x, aabb_max_y, aabb_max_z).divideScalar(1000)

      // console.log(`Star array ${i}: AABB min(${aabb_min_x}, ${aabb_min_y}, ${aabb_min_z}) max(${aabb_max_x}, ${aabb_max_y}, ${aabb_max_z})`)
      let starArr = new Float32Array(arr.buffer)

      const geometry = new BufferGeometry()
      geometry.setAttribute('position', new BufferAttribute(starArr, 3))

      const colorA = uniform(color('#246acb'));

      const material = new PointsNodeMaterial({
        blending: AdditiveBlending,
        depthWrite: false, depthTest: false,
        colorNode: vec4(colorA, 1),
      });

      // mesh
      const mesh = new Points(geometry, material);
      mesh.frustumCulled = false
      this.scene.add(mesh);
      this.requestRenderIfNotRequested()
    }
  }


  private onPointerDown = (e: PointerEvent) => {
    this.pointerDownPos.set(e.clientX, e.clientY);
    this.isDragging = false;
  };

  private onPointerMove = (e: PointerEvent) => {
    if (e.buttons === 0) return;
    const dx = e.clientX - this.pointerDownPos.x;
    const dy = e.clientY - this.pointerDownPos.y;
    if (dx * dx + dy * dy > 25) this.isDragging = true;
  };

  private onPointerUp = (e: PointerEvent) => {
    if (this.isDragging) return;

    const ndc = new Vector2(
      (e.clientX / window.innerWidth) * 2 - 1,
      -(e.clientY / window.innerHeight) * 2 + 1,
    );

    this.raycaster.setFromCamera(ndc, this.camera);

    const routeCoords = this.routeLine.getHitSpriteCoords(this.raycaster);
    if (routeCoords) {
      const { x, y, z } = routeCoords;
      const name = this.wasmModule?.get_star_from_coords(x, y, z);
      if (name) {
        this.onSystemFocus?.({ name, coords: routeCoords });
        return;
      }
    }

    const fovRad = this.camera.fov * (Math.PI / 180);
    const angularTolerance = (fovRad / window.innerHeight) * 8;
    const { origin, direction } = this.raycaster.ray;

    const hit = this.wasmModule?.find_star_near_ray(
      origin.x, origin.y, origin.z,
      direction.x, direction.y, direction.z,
      angularTolerance,
    ) as StarSystem | null;

    if (hit) {
      this.onSystemFocus?.(hit);
    }
  }
}
