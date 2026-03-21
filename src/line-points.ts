import * as THREE from 'three/webgpu';
import {
  float,
  instancedBufferAttribute,
  shapeCircle,
  uniform,
  vec4,
} from 'three/tsl';

export class LinePoints extends THREE.Object3D {
  private line: THREE.Line;
  private lineGeometry: THREE.BufferGeometry;

  private sprite: THREE.Sprite;
  private positionAttribute: THREE.InstancedBufferAttribute;
  private colorAttribute: THREE.InstancedBufferAttribute;
  private currentProgress = 0; // Index of the current progress along the route
  private currentPointCount = 0;

  private maxPoints: number;

  // For raycasting to detect clicks on the route points
  private pickPoints: THREE.Points; 
  private pickGeometry: THREE.BufferGeometry;
  

  constructor(
    initialPointsCount: number,
    color: THREE.ColorRepresentation = 0xffffff,
    spriteSize: number
  ) {
    super();

    this.maxPoints = initialPointsCount;

    /* ---------- LINE ---------- */

    this.lineGeometry = new THREE.BufferGeometry();
    this.lineGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([]), 3)
    );

    const lineMaterial = new THREE.LineBasicMaterial({ color, vertexColors: true });
    this.line = new THREE.Line(this.lineGeometry, lineMaterial);
    this.add(this.line);

    /* ---------- INSTANCED SPRITE ---------- */

    this.positionAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(initialPointsCount * 3),
      3
    );

    this.colorAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(initialPointsCount * 3),
      3
    );

    const spriteMaterial = new THREE.SpriteNodeMaterial({
      colorNode: vec4(instancedBufferAttribute(this.colorAttribute), float(1.0)),
      sizeAttenuation: false,
      maskNode: shapeCircle(),
      scaleNode: uniform(spriteSize),
      positionNode: instancedBufferAttribute(this.positionAttribute),
    });

    this.sprite = new THREE.Sprite(spriteMaterial);
    this.sprite.count = 0;
    this.sprite.frustumCulled = false;

    this.add(this.sprite);

    /* ---------- PICK POINTS (RAYCAST ONLY) ---------- */

    this.pickGeometry = new THREE.BufferGeometry();
    this.pickGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([]), 3)
    );

    const pickMaterial = new THREE.PointsMaterial({
      size: 12,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });

    this.pickPoints = new THREE.Points(this.pickGeometry, pickMaterial);
    this.pickPoints.frustumCulled = false;
    this.add(this.pickPoints);

    /* ---------- INIT ---------- */

    this.update(new Float32Array([]));
  }

  /* ---------- UPDATE ---------- */

  setProgress(index: number): void {
    this.currentProgress = index;
    this.applyPointStyles(this.currentPointCount);
  }

  getHitSpriteCoords(raycaster: THREE.Raycaster): { x: number; y: number; z: number } | null {
    const previousPointThreshold = raycaster.params.Points.threshold;
    raycaster.params.Points.threshold = 0.1;

    const hit = raycaster.intersectObject(this.pickPoints, false)[0];
    raycaster.params.Points.threshold = previousPointThreshold;

    if (!hit || typeof hit.index !== "number") {
      return null;
    }

    const spriteIndex = hit.index;
    if (spriteIndex < 0 || spriteIndex >= this.currentPointCount) {
      return null;
    }

    const offset = spriteIndex * 3;
    const positions = this.positionAttribute.array as Float32Array;
    return {
      x: positions[offset],
      y: positions[offset + 1],
      z: positions[offset + 2],
    };
  }

  private applyPointStyles(count: number): void {
    const colorArray = this.colorAttribute.array as Float32Array;
    const lineColorArray = new Float32Array(count * 3);

    for (let i = 0; i < count; i += 1) {
      const offset = i * 3;
      const isChecked = i <= this.currentProgress;

      if (isChecked) {
        // Highlight checked route nodes in red.
        colorArray[offset] = 1.0;
        colorArray[offset + 1] = 0.2;
        colorArray[offset + 2] = 0.2;
      } else {
        colorArray[offset] = 1.0;
        colorArray[offset + 1] = 1.0;
        colorArray[offset + 2] = 1.0;
      }

      lineColorArray[offset] = colorArray[offset];
      lineColorArray[offset + 1] = colorArray[offset + 1];
      lineColorArray[offset + 2] = colorArray[offset + 2];
    }

    this.colorAttribute.needsUpdate = true;

    this.lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColorArray, 3));
  }

  update(points: Float32Array): void {
    const count = points.length / 3;

    if (count > this.maxPoints) {
      // Double the buffer size
      const newMaxPoints = Math.max(this.maxPoints * 2, count);
      
      // Remove old sprite
      this.remove(this.sprite);
      
      // Create new position attribute with larger buffer
      this.positionAttribute = new THREE.InstancedBufferAttribute(
        new Float32Array(newMaxPoints * 3),
        3
      );

      this.colorAttribute = new THREE.InstancedBufferAttribute(
        new Float32Array(newMaxPoints * 3),
        3
      );
      
      // Create new sprite with the new buffer
      const spriteMaterial = new THREE.SpriteNodeMaterial({
        colorNode: vec4(instancedBufferAttribute(this.colorAttribute), float(1.0)),
        sizeAttenuation: false,
        maskNode: shapeCircle(),
        scaleNode: (this.sprite.material as THREE.SpriteNodeMaterial).scaleNode,
        positionNode: instancedBufferAttribute(this.positionAttribute),
      });
      
      this.sprite = new THREE.Sprite(spriteMaterial);
      this.sprite.count = 0;
      this.sprite.frustumCulled = false;
      this.add(this.sprite);
      
      this.maxPoints = newMaxPoints;
    }

    /* Update line */
    this.lineGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(points, 3)
    );
    this.lineGeometry.computeBoundingSphere();

    this.pickGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(points, 3)
    );
    this.pickGeometry.computeBoundingSphere();

    /* Update instanced sprite positions */
    this.positionAttribute.array.set(points);
    this.positionAttribute.needsUpdate = true;

    this.currentPointCount = count;
    this.applyPointStyles(count);

    this.sprite.count = count;
  }
}
