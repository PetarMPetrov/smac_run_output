
  (function() {
    var fn = function() {
      Bokeh.safely(function() {
        (function(root) {
          function embed_document(root) {
            
          var docs_json = '{"f05f4331-b36c-4095-af48-429bb39b907e":{"roots":{"references":[{"attributes":{"callback":null,"data":{"LPI":["00nan +/- nan"],"Parameters":["C"],"fANOVA":["100.00 +/- 0.00"]},"selected":{"id":"1013","type":"Selection"},"selection_policy":{"id":"1014","type":"UnionRenderers"}},"id":"1001","type":"ColumnDataSource"},{"attributes":{},"id":"1013","type":"Selection"},{"attributes":{},"id":"1012","type":"StringEditor"},{"attributes":{"default_sort":"descending","editor":{"id":"1012","type":"StringEditor"},"field":"LPI","formatter":{"id":"1011","type":"StringFormatter"},"title":"LPI","width":100},"id":"1004","type":"TableColumn"},{"attributes":{"default_sort":"descending","editor":{"id":"1008","type":"StringEditor"},"field":"Parameters","formatter":{"id":"1007","type":"StringFormatter"},"sortable":false,"title":"Parameters","width":150},"id":"1002","type":"TableColumn"},{"attributes":{},"id":"1014","type":"UnionRenderers"},{"attributes":{},"id":"1009","type":"StringFormatter"},{"attributes":{"source":{"id":"1001","type":"ColumnDataSource"}},"id":"1006","type":"CDSView"},{"attributes":{},"id":"1008","type":"StringEditor"},{"attributes":{"columns":[{"id":"1002","type":"TableColumn"},{"id":"1003","type":"TableColumn"},{"id":"1004","type":"TableColumn"}],"height":50,"index_position":null,"source":{"id":"1001","type":"ColumnDataSource"},"view":{"id":"1006","type":"CDSView"}},"id":"1005","type":"DataTable"},{"attributes":{},"id":"1010","type":"StringEditor"},{"attributes":{},"id":"1011","type":"StringFormatter"},{"attributes":{},"id":"1007","type":"StringFormatter"},{"attributes":{"default_sort":"descending","editor":{"id":"1010","type":"StringEditor"},"field":"fANOVA","formatter":{"id":"1009","type":"StringFormatter"},"title":"fANOVA","width":100},"id":"1003","type":"TableColumn"}],"root_ids":["1005"]},"title":"Bokeh Application","version":"1.1.0"}}';
          var render_items = [{"docid":"f05f4331-b36c-4095-af48-429bb39b907e","roots":{"1005":"a5508bb4-a4d7-47da-8dac-22d61e67ad7e"}}];
          root.Bokeh.embed.embed_items(docs_json, render_items);
        
          }
          if (root.Bokeh !== undefined) {
            embed_document(root);
          } else {
            var attempts = 0;
            var timer = setInterval(function(root) {
              if (root.Bokeh !== undefined) {
                embed_document(root);
                clearInterval(timer);
              }
              attempts++;
              if (attempts > 100) {
                console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                clearInterval(timer);
              }
            }, 10, root)
          }
        })(window);
      });
    };
    if (document.readyState != "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  })();
