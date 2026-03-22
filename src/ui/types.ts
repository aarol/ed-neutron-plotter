export interface RouteConfig {
  from: string;
  to: string;
  alreadySupercharged: boolean;
}

export interface StarSystem {
  name: string;
  coords: {
    x: number;
    y: number;
    z: number;
  };
}
