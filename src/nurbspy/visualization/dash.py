"""Host-owned Dash adapters for the shared geometry viewport and optional editor."""

from copy import deepcopy
import json
from pathlib import Path
from .model import _GeometryModel


class GeometryView(_GeometryModel):
    """Read-only component with navigation, display options, and diagnostics.

    The host owns its app and server. GeometryEditor adds editing to this same
    component; both use the desktop renderer and Python geometry model.
    """

    _editable = False

    @property
    def read_only(self):
        return not self._editable

    @property
    def layout(self):
        """Mount only stores/capabilities required by this component mode."""
        try:
            from dash import dcc, html
        except ImportError as error:
            raise ImportError(
                "Install Dash with: pip install 'nurbspy[web]'"
            ) from error
        scene = deepcopy(self._initial_scene)
        if self.read_only:
            scene.pop("edit_store", None)
        children = [
            html.Div(id=self.viewport_id, style={"height": self.height, "minWidth": 0})
        ]
        if not self.read_only:
            children.append(dcc.Store(id=self.edit_id))
        children.extend(
            [
                dcc.Store(id=self.scene_id, data=scene),
                dcc.Store(id=self.state_id, data=self.initial_state),
                dcc.Store(id=self.ready_id),
            ]
        )
        return html.Div(children, id=self.id, style={"minWidth": 0, "width": "100%"})

    def register(self, app):
        """Attach the shared viewer and only the endpoints allowed by its mode."""
        from dash import Input, Output, State
        from dash.exceptions import PreventUpdate
        from flask import Response, jsonify, request, send_from_directory

        registered = getattr(app, "_nurbspy_explorer_ids", set())
        if self.id in registered:
            raise ValueError(f"An explorer with id {self.id!r} is already registered")
        route_prefix = app.config.routes_pathname_prefix.rstrip("/")
        if not registered:
            assets = Path(__file__).parents[1] / "web_assets"
            app.server.add_url_rule(
                route_prefix + "/_nurbspy_web/<path:filename>",
                endpoint=f"nurbspy_web_assets_{id(app)}",
                view_func=lambda filename: send_from_directory(assets, filename),
            )
        inspect_path = f"/_nurbspy_web/inspect/{self.id}"

        def inspect_geometry():
            try:
                return jsonify(self.inspect(request.get_json(silent=True)))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                return jsonify(error=str(error)), 422

        app.server.add_url_rule(
            route_prefix + inspect_path,
            endpoint=f"nurbspy_inspect_{id(app)}_{self.id}",
            view_func=inspect_geometry,
            methods=["POST"],
        )
        options = {
            "readOnly": self.read_only,
            "inspectEndpoint": app.get_relative_path(inspect_path),
        }
        if not self.read_only:
            edit_path = f"/_nurbspy_web/edit/{self.id}"

            def render_edit():
                try:
                    positions, _, quality, revision = self._render_update(
                        request.get_json(silent=True)
                    )
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    return jsonify(error=str(error)), 422
                return Response(
                    positions.tobytes(),
                    mimetype="application/octet-stream",
                    headers={
                        "X-Nurbspy-Revision": str(revision),
                        "X-Nurbspy-Quality": quality,
                        "Cache-Control": "no-store",
                    },
                )

            app.server.add_url_rule(
                route_prefix + edit_path,
                endpoint=f"nurbspy_edit_{id(app)}_{self.id}",
                view_func=render_edit,
                methods=["POST"],
            )
            options.update(
                endpoint=app.get_relative_path(edit_path), stateStore=self.state_id
            )

            @app.callback(
                Output(self.scene_id, "data"),
                Output(self.state_id, "data"),
                Input(self.edit_id, "data"),
                State(self.state_id, "data"),
                prevent_initial_call=True,
            )
            def accept_edit(request, current_state):
                if request is None:
                    raise PreventUpdate
                return self.process_edit(request, current_state)

        app._nurbspy_explorer_ids = registered | {self.id}
        module_url = app.get_relative_path("/_nurbspy_web/explorer.js")
        app.clientside_callback(
            "async function(scene) {"
            f"const viewer = await import({json.dumps(module_url)});"
            f"return viewer.update({json.dumps(self.viewport_id)}, scene, {json.dumps(options)});"
            "}",
            Output(self.ready_id, "data"),
            Input(self.scene_id, "data"),
        )
        return self


class GeometryEditor(GeometryView):
    """The shared viewer plus edit controls, drag/weight edits, and accepted state.

    Add editor.layout to the app, call editor.register(app), and subscribe to
    editor.state_id for full accepted models. Input geometry remains unchanged.
    """

    _editable = True
