import numpy as np
import warnings
from shap.plots import colors
from shap.utils import ordinal_str
from shap.plots._text import (
    process_shap_values,
    unpack_shap_explanation_contents,
    svg_force_plot,
)
import random
import string
import json

try:
    from IPython.core.display import display as ipython_display, HTML

    have_ipython = True
except ImportError:
    have_ipython = False

"""
NOTE: This file is a modified version of the shap plot function. The bits that are 
different are surrounded by the comment: # Changed from shap
"""


# TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
def text(
    shap_values,
    linebreak_before_idxs,  # Added this parameter
    text_cols,  # Added this parameter
    num_starting_labels=0,
    grouping_threshold=0.01,
    separator="",
    xmin=None,
    xmax=None,
    cmax=None,
    display=True,
):
    """Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.

    Exactly the same as the shap.plots.text() function, but with the addition of linebreak_after_idxs and text_cls parameter.
    These are used to add linebreaks with the text column name at the end of the function.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap values for a string(# input_tokens x output_tokens).

    num_starting_labels : int
        Number of tokens (sorted in decending order by corresponding SHAP values) that are uncovered in the initial view. When set to 0 all tokens
        covered.

    grouping_threshold : float
        If the component substring effects are less than a grouping_threshold fraction of an unlowered interaction effect then we
        visualize the entire group as a single chunk. This is primarily used for explanations that were computed with fixed_context set to 1 or 0
        when using the Partition explainer, since this causes interaction effects to be left on internal nodes rather than lowered.

    separator : string
        The string seperator that joins tokens grouped by interation effects and unbroken string spans.

    xmin : float
        Minimum shap value bound.

    xmax : float
        Maximum shap value bound.

    cmax : float
        Maximum absolute shap value for sample. Used for scaling colors for input tokens.

    display: bool
        Whether to display or return html to further manipulate or embed. default: True

    linebreak_after_idx: list
        After how many features to add a line break, designed to split tabular and text features. This is the only part of the file
        that is different from the original shap text file. default: None

    text_cols: list
        List of column names that are text features, to be printed as col = ... default: None

    """

    def values_min_max(values, base_values):
        """Used to pick our axis limits."""
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2 and (
        shap_values.output_names is None or isinstance(shap_values.output_names, str)
    ):
        xmin = 0
        xmax = 0
        cmax = 0

        for i, v in enumerate(shap_values):
            values, clustering = unpack_shap_explanation_contents(v)
            tokens, values, group_sizes = process_shap_values(
                v.data, values, grouping_threshold, separator, clustering
            )

            if i == 0:
                xmin, xmax, cmax = values_min_max(values, v.base_values)
                continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, v.base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
    <br>
    <hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
    <div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
    </div>
                """
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
                linebreak_before_idxs=linebreak_before_idxs,
                text_cols=text_cols,
            )
        if display:
            ipython_display(HTML(out))
            return
        else:
            return out

    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
            tokens, values, group_sizes = process_shap_values(
                shap_values[:, i].data,
                values,
                grouping_threshold,
                separator,
                clustering,
            )

            # if i == 0:
            #     xmin, xmax, cmax = values_min_max(values, shap_values[:,i].base_values)
            #     continue

            xmin_i, xmax_i, cmax_i = values_min_max(
                values, shap_values[:, i].base_values
            )
            if xmin_computed is None or xmin_i < xmin_computed:
                xmin_computed = xmin_i
            if xmax_computed is None or xmax_i > xmax_computed:
                xmax_computed = xmax_i
            if cmax_computed is None or cmax_i > cmax_computed:
                cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = f"""<div align='center'>
<script>
    document._hover_{uuid} = '_tp_{uuid}_output_0';
    document._zoom_{uuid} = undefined;
    function _output_onclick_{uuid}(i) {{
        var next_id = undefined;
        
        if (document._zoom_{uuid} !== undefined) {{
            document.getElementById(document._zoom_{uuid}+ '_zoom').style.display = 'none';
            
            if (document._zoom_{uuid} === '_tp_{uuid}_output_' + i) {{
                document.getElementById(document._zoom_{uuid}).style.display = 'block';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = '3px solid #000000';
            }} else {{
                document.getElementById(document._zoom_{uuid}).style.display = 'none';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = 'none';
            }}
        }}
        if (document._zoom_{uuid} !== '_tp_{uuid}_output_' + i) {{
            next_id = '_tp_{uuid}_output_' + i;
            document.getElementById(next_id).style.display = 'none';
            document.getElementById(next_id + '_zoom').style.display = 'block';
            document.getElementById(next_id+'_name').style.borderBottom = '3px solid #000000';
        }}
        document._zoom_{uuid} = next_id;
    }}
    function _output_onmouseover_{uuid}(i, el) {{
        if (document._zoom_{uuid} !== undefined) {{ return; }}
        if (document._hover_{uuid} !== undefined) {{
            document.getElementById(document._hover_{uuid} + '_name').style.borderBottom = 'none';
            document.getElementById(document._hover_{uuid}).style.display = 'none';
        }}
        document.getElementById('_tp_{uuid}_output_' + i).style.display = 'block';
        el.style.borderBottom = '3px solid #000000';
        document._hover_{uuid} = '_tp_{uuid}_output_' + i;
    }}
</script>
<div style=\"color: rgb(120,120,120); font-size: 12px;\">outputs</div>"""
        output_values = shap_values.values.sum(0) + shap_values.base_values
        output_max = np.max(np.abs(output_values))
        for i, name in enumerate(shap_values.output_names):
            scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
            color = colors.red_transparent_blue(scaled_value)
            color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
            # '#dddddd' if i == 0 else '#ffffff' border-bottom: {'3px solid #000000' if i == 0 else 'none'};
            out += f"""
<div style="display: inline; border-bottom: {'3px solid #000000' if i == 0 else 'none'}; background: rgba{color}; border-radius: 3px; padding: 0px" id="_tp_{uuid}_output_{i}_name"
    onclick="_output_onclick_{uuid}({i})"
    onmouseover="_output_onmouseover_{uuid}({i}, this);">{name}</div>"""
        out += "<br><br>"
        for i, name in enumerate(shap_values.output_names):
            out += f"<div id='_tp_{uuid}_output_{i}' style='display: {'block' if i == 0 else 'none'}';>"
            out += text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
                linebreak_before_idxs=linebreak_before_idxs,
                text_cols=text_cols,
            )
            out += "</div>"
            out += f"<div id='_tp_{uuid}_output_{i}_zoom' style='display: none;'>"
            out += text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                display=False,
                linebreak_before_idxs=linebreak_before_idxs,
                text_cols=text_cols,
            )
            out += "</div>"
        out += "</div>"
        if display:
            ipython_display(HTML(out))
            return
        else:
            return out
        # text_to_text(shap_values)
        # return

    if len(shap_values.shape) == 3:
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            for j in range(shap_values.shape[0]):
                values, clustering = unpack_shap_explanation_contents(
                    shap_values[j, :, i]
                )
                tokens, values, group_sizes = process_shap_values(
                    shap_values[j, :, i].data,
                    values,
                    grouping_threshold,
                    separator,
                    clustering,
                )

                xmin_i, xmax_i, cmax_i = values_min_max(
                    values, shap_values[j, :, i].base_values
                )
                if xmin_computed is None or xmin_i < xmin_computed:
                    xmin_computed = xmin_i
                if xmax_computed is None or xmax_i > xmax_computed:
                    xmax_computed = xmax_i
                if cmax_computed is None or cmax_i > cmax_computed:
                    cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
<br>
<hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
<div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
</div>
            """
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
                linebreak_before_idxs=linebreak_before_idxs,
                text_cols=text_cols,
            )
        if display:
            ipython_display(HTML(out))
            return
        else:
            return out

    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(
        shap_values.values, shap_values.base_values
    )
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new

    values, clustering = unpack_shap_explanation_contents(shap_values)
    # tokens, values, group_sizes = process_shap_values(
    #     shap_values.data, values, grouping_threshold, separator, clustering,
    # )

    (
        tokens,
        values,
        group_sizes,
        token_id_to_node_id_mapping,
        collapsed_node_ids,
    ) = process_shap_values(
        shap_values.data,
        values,
        grouping_threshold,
        separator,
        clustering,
        return_meta_data=True,
    )

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    maxv = values.max()
    minv = values.min()
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())

    # uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [
        t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "") for t in tokens
    ]
    output_name = (
        shap_values.output_names if isinstance(shap_values.output_names, str) else ""
    )
    out += svg_force_plot(
        values,
        shap_values.base_values,
        shap_values.base_values + values.sum(),
        encoded_tokens,
        uuid,
        xmin,
        xmax,
        output_name,
    )

    # Changed from shap
    ################################
    if linebreak_before_idxs is not None:
        lines_inserted = 0
        linebreak_idxs = []
        for idx in linebreak_before_idxs:
            cluster_idx = np.where(
                collapsed_node_ids == token_id_to_node_id_mapping[idx]
            )[0][0]
            tokens = np.insert(tokens, cluster_idx + lines_inserted, "")
            values = np.insert(values, cluster_idx + lines_inserted, 0)
            group_sizes = np.insert(group_sizes, cluster_idx + lines_inserted, 1)
            linebreak_idxs.append(cluster_idx + lines_inserted)
            lines_inserted += 1

    ################################
    out += "<div align='center'><div style=\"color: rgb(120,120,120); font-size: 12px; margin-top: -15px;\">inputs</div>"
    text_col_count = 0

    for i, token in enumerate(tokens):
        # Changed from shap
        ################################
        if linebreak_idxs is not None:
            if i in linebreak_idxs:
                token = f"<br>(Text ft) {text_cols[text_col_count]} = "
                # token = f"HELLO__{i, display}"
                # we add a line break between the tabular features and the text features
                out += f"""<div style='display: inline; text-align: center;'
            ><div style='display: none; color: #999; padding-top: 0px; font-size: 12px;'></div
                ><div id='_tp_{uuid}_ind_newline'
                    style='display: inline; border-radius: 3px; padding: 0px'
                >{token}</div></div>"""
                text_col_count += 1
                continue
            else:
                element_id = i - text_col_count
        else:
            element_id = i
        ################################

        scaled_value = 0.5 + 0.5 * values[i] / (cmax + 1e-8)
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"

        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])

        # Changed from shap: element_id used instead of i, nothing else changed
        #################################
        # the HTML for this token
        out += f"""<div style='display: {wrapper_display}; text-align: center;'
        ><div style='display: {label_display}; color: #999; padding-top: 0px; font-size: 12px;'>{value_label}</div
            ><div id='_tp_{uuid}_ind_{element_id}'
                style='display: inline; background: rgba{color}; border-radius: 3px; padding: 0px'
                onclick="
                if (this.previousSibling.style.display == 'none') {{
                    this.previousSibling.style.display = 'block';
                    this.parentNode.style.display = 'inline-block';
                }} else {{
                    this.previousSibling.style.display = 'none';
                    this.parentNode.style.display = 'inline';
                }}"
                onmouseover="document.getElementById('_fb_{uuid}_ind_{element_id}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{element_id}').style.opacity = 1;"
                onmouseout="document.getElementById('_fb_{uuid}_ind_{element_id}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{element_id}').style.opacity = 0;"
            >{token.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '')}</div></div>"""
        #################################
    out += "</div>"

    if display:
        ipython_display(HTML(out))
        return
    else:
        return out
