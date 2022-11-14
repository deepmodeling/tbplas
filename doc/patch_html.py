#! /usr/bin/env python

import os


def patch_stat(content):
    patch = """
    <script>
        var _hmt = _hmt || [];
        (function() {
        var hm = document.createElement("script");
        hm.src = "https://hm.baidu.com/hm.js?b04150564935fd379566e6a5a255f1d7";
        var s = document.getElementsByTagName("script")[0]; 
        s.parentNode.insertBefore(hm, s);
        })();
    </script>\n\n"""
    try:
        nl_head = content.index("</head>\n")
    except ValueError:
        nl_head = None
    if nl_head is not None:
        content.insert(nl_head, patch)


def patch_icp(content):
    icp = """
    <div align=center>
        <br>
            <a href="http://beian.miit.gov.cn/"; target=_blank>鄂ICP备2022004007号</a>
        </br>
    </div>\n\n"""
    nl_footer = content.index("</footer>\n")
    content.insert(nl_footer-1, icp)


def main():
    html_files = os.popen("find build/html -name '*.html'").readlines()
    for html in html_files:
        file_name = html.rstrip("\n")
        with open(file_name, "r") as f:
            content = f.readlines()
            patch_stat(content)
            patch_icp(content)
        with open(file_name, "w") as f:
            f.writelines(content)


if __name__ == "__main__":
    main()
