# coding=utf-8
import streamlit as st

st.set_page_config(
    page_title="期权监控平台",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={})


def create_side_settings(st):
    with st.sidebar:
        st.write('日度期权监控-Settings')

        c11, c12 = st.columns([2, 1])
        op_api_base = c11.selectbox("期权行情API接口", ['47.104.186.157', ], index=0, )
        op_api_port = c12.selectbox("端口", ['3100', ], index=0, )

        c21, c22 = st.columns([2, 1])
        idx_api_base = c21.selectbox("指数行情API接口", ['47.104.186.157', ], index=0, placeholder="指数行情API接口",
                                     )
        idx_api_port = c22.selectbox("端口", ['3100', ], index=0, placeholder="指数行情API端口", )

        c31, c32 = st.columns([2, 1])
        greek_api_base = c31.selectbox("Greek API接口", ['47.104.186.157', ], index=0, placeholder="Greek API接口", )
        greek_api_port = c32.selectbox("端口", ['3100', ], index=0, placeholder="Greek API端口", )

        c41, c42 = st.columns([2, 1])
        optimize_api_base = c41.selectbox("optimization API接口", ['47.104.186.157', ], index=0,
                                          placeholder="optimization API接口",
                                          )
        optimize_api_port = c42.selectbox("端口", ['3100', ], index=0, placeholder="optimization API端口", )

        return op_api_base + ':' + op_api_port, idx_api_base + ':' + idx_api_port, greek_api_base + ':' + greek_api_port, optimize_api_base + ':' + optimize_api_port


if __name__ == '__main__':
    op_api_base, idx_api_base, greek_api_base, optimize_api_base = create_side_settings(st)

    ## 行情显示

    ## 持仓显示


    ## greek显示


    ## 推荐组合显示

    pass
