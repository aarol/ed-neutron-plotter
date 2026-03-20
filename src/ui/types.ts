export interface RouteConfig {
  from: string;
  to: string;
  alreadySupercharged: boolean;
}

export interface RouteNode {
  name: string;
  coords: {
    x: number;
    y: number;
    z: number;
  };
}

export interface TargetInfoState {
  name: string;
  x: number;
  y: number;
  z: number;
}