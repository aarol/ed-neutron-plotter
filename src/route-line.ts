import { BufferAttribute, BufferGeometry, Line, Scene, Vector3 } from "three";
import { LineBasicNodeMaterial } from "three/webgpu";

export class RouteLine {
  material = new LineBasicNodeMaterial({ color: 0xffffff });
  line: Line;

  constructor(scene: Scene) {
    const geometry = new BufferGeometry();
    const positions = new Float32Array(0);
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    this.line = new Line(geometry, this.material);
    scene.add(this.line);
  }

  updatePoints(points: Vector3[]) {
    console.log("Updating route line points:", points);
    const positions = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      positions[i * 3] = points[i].x;
      positions[i * 3 + 1] = points[i].y;
      positions[i * 3 + 2] = points[i].z;
    }
    this.line.geometry.setAttribute('position', new BufferAttribute(positions, 3));
    this.line.geometry.computeBoundingSphere();
  }
}