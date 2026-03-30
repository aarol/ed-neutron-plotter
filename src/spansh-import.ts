import { api } from "./api";
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
      return {
        system: {
          name: cells[colIndices["System Name"]]?.textContent?.trim() || "",
        },
        distance: parseFloat(cells[colIndices["Distance (LY)"]]?.textContent?.trim() || "0"),
        refuel: cells[colIndices["Refuel?"]]?.textContent?.trim() === "Yes",
        isNeutron: cells[colIndices["Neutron"]]?.textContent?.trim() === "Yes",
      }
    })

    const systemNames = nodes.map(node => node.system.name)

    const systemToCoordsMap = await api.getMultipleStarCoordsFromApi(systemNames);
    return nodes.map(node => {
      const name = node.system.name;
      const coords = systemToCoordsMap[name];
      if (!coords) {
        throw new Error(`Coordinates not found for system: ${node.system.name}`);
      }
      return {
        ...node,
        system: { name, coords }
      }
    })

  } catch (error) {
    throw new Error(`Failed to parse Spansh HTML. Please ensure you are copying the whole page from Spansh.
If the problem persists, the HTML structure may have changed. Please report this issue with the HTML content that caused the error. Original error: ${error}
      `);
  }
}

