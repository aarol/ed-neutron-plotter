import eddb from "./api/eddb";
import spansh from "./api/spansh";
import type { RouteNode } from "./ui/state/routeModel";


export async function parseSpanshHtml(html: string): Promise<RouteNode[]> {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, "text/html");
  const table = doc.querySelector("table");

  const headers = Array.from(table?.querySelectorAll("thead td") || []).map(th => th.textContent?.trim() || "");

  try {
    const colIndices = {
      "System Name": headers.indexOf("System Name"),
      "Distance (LY)": headers.indexOf("Distance (LY)"),
      "Refuel?": headers.indexOf("Refuel?"),
      "Neutron": headers.indexOf("Neutron"),
    };

    const rows = Array.from(table?.querySelectorAll("tbody tr") || []);
    const nodes = rows.map(row => {
      const cells = row.querySelectorAll("td");
      const system = cells[colIndices["System Name"]].querySelector("a")?.getAttribute("href")?.split("/system/")[1];
      return {
        system: {
          name: cells[colIndices["System Name"]]?.textContent?.trim() || "",
          id64: system ? parseInt(system) : null
        },
        distance: parseFloat(cells[colIndices["Distance (LY)"]]?.textContent?.trim() || "0"),
        refuel: cells[colIndices["Refuel?"]]?.textContent?.trim() === "Yes",
        isNeutron: cells[colIndices["Neutron"]]?.textContent?.trim() === "Yes",
      }
    })

    const systemNames = nodes.map(node => node.system.name)

    const systemToCoordsMap = await eddb.getMultipleStarCoordsFromApi(systemNames);

    const route: RouteNode[] = [];

    for (const node of nodes) {
      const name = node.system.name;
      let coords = systemToCoordsMap[name];
      if (!coords) {
        console.warn(`Coordinates not found for system: ${node.system.name}`);
        if (!node.system.id64) {
          throw new Error(`No ID64 for system: ${node.system.name}`);
        }
        const coordsFromSpansh = await spansh.getSystemCoords(node.system.id64)
        if (!coordsFromSpansh) {
          throw new Error(`Coordinates not found from Spansh for system: ${node.system.name} with ID64: ${node.system.id64}`);
        }
        console.info(`Got coordinates from Spansh for system ${node.system.name}: (${coordsFromSpansh.x}, ${coordsFromSpansh.y}, ${coordsFromSpansh.z})`);
        coords = coordsFromSpansh;
      }
      route.push({
        ...node,
        system: { name, coords }
      });
    }
    return route;
  } catch (error) {
    throw new Error(`Failed to parse Spansh HTML. Please ensure you are copying the whole page from Spansh.
If the problem persists, the HTML structure may have changed. Please report this issue with the HTML content that caused the error. Original error: ${error}
      `);
  }
}

