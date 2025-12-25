import * as THREE from 'three/webgpu';
import {
  instancedBufferAttribute,
  shapeCircle,
  uniform
} from 'three/tsl';

export class LinePoints extends THREE.Object3D {
  private line: THREE.Line;
  private lineGeometry: THREE.BufferGeometry;

  private sprite: THREE.Sprite;
  private positionAttribute: THREE.InstancedBufferAttribute;

  private maxPoints: number;

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

    const lineMaterial = new THREE.LineBasicMaterial({ color });
    this.line = new THREE.Line(this.lineGeometry, lineMaterial);
    this.add(this.line);

    /* ---------- INSTANCED SPRITE ---------- */

    this.positionAttribute = new THREE.InstancedBufferAttribute(
      new Float32Array(initialPointsCount * 3),
      3
    );

    const spriteMaterial = new THREE.SpriteNodeMaterial({
      color: color,
      sizeAttenuation: false,
      maskNode: shapeCircle(),
      scaleNode: uniform(spriteSize),
      positionNode: instancedBufferAttribute(this.positionAttribute),
    });

    this.sprite = new THREE.Sprite(spriteMaterial);
    this.sprite.count = 0;
    this.sprite.frustumCulled = false;

    this.add(this.sprite);

    /* ---------- INIT ---------- */

    this.update(new Float32Array([]));
  }

  /* ---------- UPDATE ---------- */

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
      
      // Create new sprite with the new buffer
      const spriteMaterial = new THREE.SpriteNodeMaterial({
        color: (this.sprite.material as THREE.SpriteNodeMaterial).color,
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

    /* Update instanced sprite positions */
    this.positionAttribute.array.set(points);
    this.positionAttribute.needsUpdate = true;

    this.sprite.count = count;
  }
}
