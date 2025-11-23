"""
Enhanced Real-Time Vehicle Feed table with Email and SMS action buttons.
"""

import streamlit as st
import pandas as pd
import html as _html
from config import config
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode


def _risk_level_from_pct(pred_pct: float) -> str:
    if pred_pct > config.risk.high_threshold:
        return "High"
    elif pred_pct >= config.risk.medium_threshold:
        return "Medium"
    return "Low"


def _is_high_risk(pred_pct: float) -> bool:
    return _risk_level_from_pct(pred_pct) == "High"

def render_enhanced_inference_table(df_log: pd.DataFrame, date_range=None, text_filter: str = "", rows_to_show: int = 50):
    """Render the enhanced Real-Time Vehicle Feed table with integrated action buttons."""
    
    # Apply filters
    # Handle different date_range formats from Streamlit
    mask = pd.Series([True] * len(df_log), index=df_log.index)
    
    if date_range is not None:
        try:
            # Handle different date_range formats from Streamlit
            # Streamlit date_input can return tuple, list, single date, or empty tuple/list during selection
            if isinstance(date_range, (tuple, list)):
                if len(date_range) == 2:
                    # Date range: (start_date, end_date)
                    dr_start, dr_end = date_range
                    if dr_start is not None and dr_end is not None:
                        mask = (df_log["timestamp"].dt.date >= dr_start) & (df_log["timestamp"].dt.date <= dr_end)
                elif len(date_range) == 1:
                    # Single date in a list/tuple (user selected only one date)
                    single_date = date_range[0]
                    if single_date is not None:
                        mask = df_log["timestamp"].dt.date == single_date
                # If empty tuple/list (len == 0), use all data (mask already set to True)
            else:
                # Single date object (not in tuple/list)
                mask = df_log["timestamp"].dt.date == date_range
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error applying date range filter: {e}")
    
    if text_filter and text_filter.strip():
        try:
            t = text_filter.strip().lower()
            text_mask = (
                df_log["model"].astype(str).str.lower().str.contains(t, na=False, regex=False) | 
                df_log["primary_failed_part"].astype(str).str.lower().str.contains(t, na=False, regex=False)
            )
            mask = mask & text_mask
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error applying text filter: {e}")
    
    df_show = df_log[mask].sort_values("timestamp", ascending=False).head(rows_to_show)
    
    if df_show.empty:
        st.markdown("<div style='padding:12px; color:#94a3b8; text-align: center;'>No data found for the selected filters.</div>", unsafe_allow_html=True)
        return

    # =================== AgGrid-based table with tooltips ===================
    df_grid = df_show.copy()

    # Remove only bucket columns from the grid to avoid blank header artifacts
    cols_to_remove = ['mileage_bucket', 'age_bucket']
    try:
        to_drop = [c for c in cols_to_remove if c in df_grid.columns]
        if to_drop:
            df_grid = df_grid.drop(columns=to_drop)
    except Exception:
        pass

    # Normalize timestamp for display
    if 'timestamp' in df_grid.columns:
        def _fmt_ts(x):
            try:
                return x if isinstance(x, str) else x.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(x)
        df_grid['timestamp'] = df_grid['timestamp'].apply(_fmt_ts)

    # Derived columns for actions visibility (email/sms text; actual click handled via selection)
    def _act_text(val: float) -> str:
        try:
            return "email" if _is_high_risk(float(val)) else "—"
        except Exception:
            return "—"
    def _act_text_sms(val: float) -> str:
        try:
            return "sms" if _is_high_risk(float(val)) else "—"
        except Exception:
            return "—"
    df_grid['email'] = df_grid['pred_prob_pct'].apply(_act_text)
    df_grid['sms'] = df_grid['pred_prob_pct'].apply(_act_text_sms)

    # Ensure column order: VIN, Event, Model, PFP, Mileage, Age, Pred %, Email, SMS, then others
    try:
        preferred = [
            'vin',
            'timestamp',
            'model',
            'primary_failed_part',
            'mileage',
            'age',
            'pred_prob_pct',
            'email',
            'sms',
        ]
        present_pref = [c for c in preferred if c in df_grid.columns]
        remaining = [c for c in df_grid.columns if c not in present_pref]
        # Keep remaining (e.g., prescriptive_summary) so tooltips can access, but hide them in grid options
        df_grid = df_grid[present_pref + remaining]
    except Exception:
        pass

    # Hide columns not for display
    for hidden_col in ['pred_prob', 'lat', 'lon', 'latitude', 'longitude']:
        if hidden_col in df_grid.columns:
            # Keep for data but will be hidden in grid options
            pass

    # Tooltip: prescriptive_summary (fallback message when empty)
    tooltip_js = JsCode("""
        function(params) {
            const row = params && params.data;
            const val = row && row.prescriptive_summary;
            if (!val || String(val).trim() === "" || String(val).toLowerCase() === "nan") {
                return "Prescriptive summary not available for this record.";
            }
            return String(val);
        }
    """)

    # Risk-based color for Pred %
    pred_style_js = JsCode(f"""
        function(params) {{
            var v = Number(params.value);
            if (isNaN(v)) return {{}};
            if (v > {config.risk.high_threshold}) {{ return {{color: '#dc2626', fontWeight: 'bold'}}; }}
            if (v >= {config.risk.medium_threshold}) {{ return {{color: '#fbbf24', fontWeight: 'bold'}}; }}
            return {{}};
        }}
    """)

    gb = GridOptionsBuilder.from_dataframe(df_grid)
    gb.configure_default_column(
        tooltipValueGetter=tooltip_js,
        sortable=True,
        filter=True,
        resizable=True
    )

    # Column headers and formatting to match prior UI
    if 'vin' in df_grid.columns:
        gb.configure_column('vin', headerName='VIN', cellStyle={'fontFamily': 'monospace'})
    if 'timestamp' in df_grid.columns:
        gb.configure_column('timestamp', headerName='Event')
    if 'model' in df_grid.columns:
        gb.configure_column('model', headerName='Model')
    if 'primary_failed_part' in df_grid.columns:
        gb.configure_column('primary_failed_part', headerName='PFP')
    if 'mileage' in df_grid.columns:
        gb.configure_column('mileage', headerName='Mileage', type=['numericColumn'])
    if 'age' in df_grid.columns:
        gb.configure_column('age', headerName='Age', type=['numericColumn'])
    if 'pred_prob_pct' in df_grid.columns:
        gb.configure_column('pred_prob_pct', headerName='Pred %', type=['numericColumn'],
                            valueFormatter="(x != null && !isNaN(x)) ? Number(x).toFixed(1) + '%' : ''",
                            cellStyle=pred_style_js)
    # Class-based renderers to draw small clickable buttons that select the row
    email_btn_renderer = JsCode(f"""
        class EmailBtnRenderer {{
            init(params) {{
                this.params = params;
                const v = Number(params.data && params.data.pred_prob_pct);
                if (isNaN(v) || !(v > {config.risk.high_threshold})) {{
                    this.eGui = document.createElement('span');
                    this.eGui.textContent = '—';
                    this.eGui.style.display = 'inline-block';
                    this.eGui.style.textAlign = 'center';
                    this.eGui.style.width = '100%';
                    return;
                }}
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = 'email';
                btn.style.padding = '1px 8px';
                btn.style.borderRadius = '6px';
                btn.style.border = '1px solid #2563eb';
                btn.style.background = '#3b82f6';
                btn.style.color = '#fff';
                btn.style.fontSize = '10px';
                btn.style.cursor = 'pointer';
                btn.style.display = 'block';
                btn.style.margin = '0 auto';
                btn.addEventListener('click', () => {{
                    const node = params.node;
                    if (node && params.api) {{
                        params.api.deselectAll();
                        node.setSelected(true);
                    }}
                    // store action hint so Python can read selected row and know intent via below buttons
                    window.__aggridLastAction = 'email';
                }});
                this.eGui = btn;
            }}
            getGui() {{ return this.eGui; }}
            refresh() {{ return false; }}
        }}
    """)

    sms_btn_renderer = JsCode(f"""
        class SmsBtnRenderer {{
            init(params) {{
                this.params = params;
                const v = Number(params.data && params.data.pred_prob_pct);
                if (isNaN(v) || !(v > {config.risk.high_threshold})) {{
                    this.eGui = document.createElement('span');
                    this.eGui.textContent = '—';
                    this.eGui.style.display = 'inline-block';
                    this.eGui.style.textAlign = 'center';
                    this.eGui.style.width = '100%';
                    return;
                }}
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = 'sms';
                btn.style.padding = '1px 8px';
                btn.style.borderRadius = '6px';
                btn.style.border = '1px solid #059669';
                btn.style.background = '#10b981';
                btn.style.color = '#0b1220';
                btn.style.fontSize = '10px';
                btn.style.cursor = 'pointer';
                btn.style.display = 'block';
                btn.style.margin = '0 auto';
                btn.addEventListener('click', () => {{
                    const node = params.node;
                    if (node && params.api) {{
                        params.api.deselectAll();
                        node.setSelected(true);
                    }}
                    window.__aggridLastAction = 'sms';
                }});
                this.eGui = btn;
            }}
            getGui() {{ return this.eGui; }}
            refresh() {{ return false; }}
        }}
    """)

    suppress_tooltip_js = JsCode("function(){ return null; }")
    if 'email' in df_grid.columns:
        gb.configure_column('email', headerName='Email',
                            sortable=False, filter=False, suppressMenu=True,
                            cellRenderer=email_btn_renderer, cellStyle={'textAlign': 'center'},
                            tooltipValueGetter=suppress_tooltip_js)
    if 'sms' in df_grid.columns:
        gb.configure_column('sms', headerName='SMS',
                            sortable=False, filter=False, suppressMenu=True,
                            cellRenderer=sms_btn_renderer, cellStyle={'textAlign': 'center'},
                            tooltipValueGetter=suppress_tooltip_js)

    # Hide non-visible columns to avoid stray empty headers
    visible_cols = set(['vin','timestamp','model','primary_failed_part','mileage','age','pred_prob_pct','email','sms'])
    always_hide = set(['pred_prob','lat','lon','latitude','longitude'])
    for c in df_grid.columns:
        if (c in always_hide) or (c not in visible_cols):
            gb.configure_column(c, hide=True)

    gb.configure_selection('single', use_checkbox=False)
    gb.configure_grid_options(enableBrowserTooltips=True, domLayout='normal', rowHeight=38, suppressRowClickSelection=False)

    grid_response = AgGrid(
        df_grid,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        data_return_mode=DataReturnMode.AS_INPUT,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="streamlit",
        height=420
    )

    # Summary statistics below grid
    high_risk_count = int(df_show['pred_prob_pct'].apply(_is_high_risk).sum())
    total_count = len(df_show)
    if high_risk_count > 0:
        st.markdown(f"""
        <div style="
            background: #fef2f2; 
            border: 1px solid #fecaca; 
            border-radius: 8px; 
            padding: 12px; 
            margin-top: 12px;
        ">
            <div style="color: #dc2626; font-weight: 600; margin-bottom: 4px;">
                High Risk Alert Summary
            </div>
            <div style="color: #374151; font-size: 14px;">
                {high_risk_count} out of {total_count} vehicles require immediate attention (Pred % > {int(config.risk.high_threshold)}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("No high-risk vehicles detected. All vehicles are within normal parameters.")

    return
    
    # Define column renaming
    column_renames = {
        "timestamp": "Event Timestamp",
        "model": "Model",
        "primary_failed_part": "Primary Failed Part",
        "mileage": "Mileage",
        "mileage_bucket": "Mileage",
        "age": "Age",
        "age_bucket": "Age",
        "pred_prob_pct": "Predictive %"
    }
    
    # Rename columns that exist
    df_display = df_show.rename(columns=column_renames)
    
    # Hide pred_prob, lat, and lon columns from table display (but keep them in data for CSV download)
    columns_to_hide = ["pred_prob", "lat", "lon", "latitude", "longitude"]
    columns_to_display = [col for col in df_display.columns if col not in columns_to_hide]
    df_display = df_display[columns_to_display]
    
    # Display the table with integrated action buttons - no spacing div
    # Add table headers with minimal spacing
    header_columns = st.columns([1.8, 0.15, 1.9, 1.4, 1.6, 1.2, 1, 1, 1, 1], gap="medium")
    header_col_vin, header_col_spacer, header_col_event, header_col_model, header_col_pfp, header_col4, header_col5, header_col6, header_col7, header_col8 = header_columns
    
    with header_col_vin:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>VIN</strong></div>', unsafe_allow_html=True)
    with header_col_event:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Event</strong></div>', unsafe_allow_html=True)
    with header_col_model:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Model</strong></div>', unsafe_allow_html=True)
    with header_col_pfp:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>PFP</strong></div>', unsafe_allow_html=True)
    with header_col4:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Mileage</strong></div>', unsafe_allow_html=True)
    with header_col5:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Age</strong></div>', unsafe_allow_html=True)
    with header_col6:
        st.markdown('<div style="margin: 0; padding: 2px 0;"><strong>Pred %</strong></div>', unsafe_allow_html=True)
    with header_col7:
        st.markdown('<div style="margin: 0; padding: 2px 0; text-align: center;"><strong>Email</strong></div>', unsafe_allow_html=True)
    with header_col8:
        st.markdown('<div style="margin: 0; padding: 2px 0; text-align: center;"><strong>SMS</strong></div>', unsafe_allow_html=True)
    
    # Add separator line with minimal spacing
    st.markdown('<div style="margin: 0; padding: 1px 0;"><hr style="border: 1px solid #374151; margin: 0;"></div>', unsafe_allow_html=True)
    
    tooltips_js = """
    <script>
    (function() {
        const state = window._rowTooltipState = window._rowTooltipState || {
            activeRow: null,
            tooltipEl: null
        };

        window.createRowTooltip = function(el, htmlContent, rowKey) {
            if (state.tooltipEl) {
                state.tooltipEl.remove();
                state.tooltipEl = null;
            }
            const tooltip = document.createElement('div');
            tooltip.className = 'row-tooltip';
            tooltip.innerHTML = htmlContent;
            tooltip.dataset.rowKey = rowKey;
            document.body.appendChild(tooltip);

            const rect = el.getBoundingClientRect();
            const top = rect.top + window.scrollY + rect.height + 6;
            let left = rect.left + 20;
            const maxLeft = window.innerWidth - tooltip.offsetWidth - 16;
            left = Math.min(left, maxLeft);

            tooltip.style.top = `${Math.max(0, top)}px`;
            tooltip.style.left = `${Math.max(16, left)}px`;

            state.tooltipEl = tooltip;
            state.activeRow = rowKey;
        };

        window.removeRowTooltip = function(rowKey) {
            if (!state.tooltipEl) return;
            if (rowKey && state.activeRow !== rowKey) return;
            state.tooltipEl.remove();
            state.tooltipEl = null;
            state.activeRow = null;
        };
    })();
    </script>
    <style>
    .row-tooltip {
        position: absolute;
        z-index: 9999;
        background: rgba(15, 23, 42, 0.96);
        color: #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        width: 320px;
        max-width: calc(100vw - 40px);
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148,163,184,0.35);
        font-size: 12px;
        line-height: 1.55;
        pointer-events: none;
    }
    .row-tooltip strong {
        color: #facc15;
    }
    .inference-row:hover {
        background: rgba(148, 163, 184, 0.08);
        border-radius: 6px;
    }
    button[kind="primary"],
    button[kind="secondary"] {
        margin: 0 auto !important;
        display: block !important;
    }
    </style>
    """
    st.markdown(tooltips_js, unsafe_allow_html=True)

    # Create action buttons for each row with reduced spacing
    for idx, (_, row) in enumerate(df_show.iterrows()):
        col_vin, spacer_col, col_event, col_model, col_pfp, col4, col5, col6, col7, col8 = st.columns(
            [1.8, 0.15, 1.9, 1.4, 1.6, 1.2, 1, 1, 1, 1], gap="medium"
        )
        
        row_key = f"inference-row-{idx}"
        summary_text = row.get("prescriptive_summary", "")
        if summary_text is None or str(summary_text).strip() in ["", "nan", "None"]:
            summary_text = "Prescriptive summary not available for this record."
        summary_html_content = _html.escape(str(summary_text)).replace("\n", "<br>")
        summary_attr = summary_html_content.replace('"', "&quot;")
        row_attr = f'data-row="{row_key}" class="inference-row" data-summary="{summary_attr}"'

        with col_vin:
            vin_display = row.get("vin", "N/A")
            st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px; font-family: monospace;">{vin_display}</div>', unsafe_allow_html=True)
        
        timestamp_val = row["timestamp"]
        if isinstance(timestamp_val, str):
            timestamp_str = timestamp_val
        else:
            timestamp_str = timestamp_val.strftime("%Y-%m-%d %H:%M:%S")

        with col_event:
            st.markdown(
                f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">'
                f'{timestamp_str}</div>',
                unsafe_allow_html=True
            )
        
        with col_model:
            st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{row["model"]}</div>', unsafe_allow_html=True)
        
        with col_pfp:
            st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{row["primary_failed_part"]}</div>', unsafe_allow_html=True)
        
        with col4:
            # Handle both 'mileage' and 'mileage_bucket' columns
            mileage_value = row.get('mileage', row.get('mileage_bucket', 'N/A'))
            if isinstance(mileage_value, (int, float)):
                st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{mileage_value:,.0f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{mileage_value}</div>', unsafe_allow_html=True)
        
        with col5:
            # Handle both 'age' and 'age_bucket' columns
            age_value = row.get('age', row.get('age_bucket', 'N/A'))
            if isinstance(age_value, (int, float)):
                st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{age_value:.1f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{age_value}</div>', unsafe_allow_html=True)
        
        with col6:
            # Color code the predictive percentage
            pred_pct = row['pred_prob_pct']
            if _is_high_risk(pred_pct):
                st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;"><span style="color: #dc2626; font-weight: bold;">{pred_pct:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div {row_attr} style="margin: 0; padding: 1px 0; font-size: 13px;">{pred_pct:.1f}%</div>', unsafe_allow_html=True)
        
        with col7:
            if _is_high_risk(pred_pct):
                email_key = f"email_{idx}_{row['timestamp']}"
                container_html = (
                    f'<div {row_attr} style="display: flex; justify-content: center; align-items: center; width: 100%;">'
                )
                st.markdown(container_html, unsafe_allow_html=True)
                if st.button("email", key=email_key, type="primary"):
                    st.info("Email functionality coming soon")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div {row_attr} style="display: flex; justify-content: center; align-items: center; width: 100%; min-height: 38.4px; margin: 0; padding: 0;">'
                    '<span style="font-size: 13px; text-align: center;">—</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
        
        with col8:
            if _is_high_risk(pred_pct):
                container_html = (
                    f'<div {row_attr} style="display: flex; justify-content: center; align-items: center; width: 100%;">'
                )
                st.markdown(container_html, unsafe_allow_html=True)
                if st.button("sms", key=f"sms_{idx}_{row['timestamp']}", type="secondary"):
                    sms_text = f"URGENT: {row['model']} vehicle has {row['pred_prob_pct']:.1f}% failure risk for {row['primary_failed_part']}. Immediate service required. Contact owner ASAP."
                    st.info("SMS text ready to copy")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div {row_attr} style="display: flex; justify-content: center; align-items: center; width: 100%; min-height: 38.4px; margin: 0; padding: 0;">'
                    '<span style="font-size: 13px; text-align: center;">—</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        """
        <script>
        (function bindRowTooltips(attempt = 0) {
            const MAX_ATTEMPTS = 60;
            const cells = document.querySelectorAll('.inference-row[data-summary]');
            console.debug('[InferenceTooltip] Attempt', attempt, '- cells detected:', cells.length);
            if (!cells.length) {
                if (attempt < MAX_ATTEMPTS) {
                    setTimeout(() => bindRowTooltips(attempt + 1), 120);
                } else {
                    console.warn('[InferenceTooltip] Gave up binding tooltips after', MAX_ATTEMPTS, 'attempts');
                }
                return;
            }
            let boundCount = 0;
            cells.forEach((cell) => {
                if (cell.dataset.tooltipBound === "true") {
                    return;
                }
                if (!cell.dataset.summary || !cell.dataset.row) {
                    return;
                }
                boundCount += 1;
                cell.dataset.tooltipBound = "true";
                cell.classList.add('row-tooltip-bound');
                cell.addEventListener("mouseenter", function(event) {
                    if (!window.createRowTooltip) {
                        console.debug('[InferenceTooltip] createRowTooltip not ready yet');
                        return;
                    }
                    window.createRowTooltip(event.currentTarget, event.currentTarget.dataset.summary, event.currentTarget.dataset.row);
                });
                cell.addEventListener("mouseleave", function(event) {
                    if (!window.removeRowTooltip) {
                        return;
                    }
                    const rowKey = event.currentTarget.dataset.row;
                    const related = event.relatedTarget;
                    if (related && related.closest && related.closest('[data-row="' + rowKey + '"]')) {
                        return;
                    }
                    window.removeRowTooltip(rowKey);
                });
            });
            console.debug('[InferenceTooltip] Newly bound cells:', boundCount);
            if (boundCount === 0 && attempt < MAX_ATTEMPTS) {
                setTimeout(() => bindRowTooltips(attempt + 1), 150);
            }
        })();
        </script>
        """,
        unsafe_allow_html=True
    )

    
    # Add summary statistics
    high_risk_count = int(df_show['pred_prob_pct'].apply(_is_high_risk).sum())
    total_count = len(df_show)
    
    if high_risk_count > 0:
        st.markdown(f"""
        <div style="
            background: #fef2f2; 
            border: 1px solid #fecaca; 
            border-radius: 8px; 
            padding: 12px; 
            margin-top: 12px;
        ">
            <div style="color: #dc2626; font-weight: 600; margin-bottom: 4px;">
                High Risk Alert Summary
            </div>
            <div style="color: #374151; font-size: 14px;">
                {high_risk_count} out of {total_count} vehicles require immediate attention (Pred % > {int(config.risk.high_threshold)}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("No high-risk vehicles detected. All vehicles are within normal parameters.")
