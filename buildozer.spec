# =============================================================================
#  buildozer.spec - Adaptador Pedagógico Inclusivo
#
#  Testado para: Buildozer 1.6.0 + python-for-android 2026.05.09 (fixado em
#  p4a.commit), Python 3.14 no Android, Kivy 2.3.1, NDK r28c, Java 17.
#  Comando: buildozer android debug   (gera bin/*.apk)
#
#  ATENÇÃO: o Buildozer não aceita comentários na mesma linha de um valor.
#  Cada comentário tem de estar numa linha própria, a começar por '#'.
# =============================================================================

[app]

# (str) Nome mostrado no Android
title = Adaptador Pedagógico Inclusivo

# (str) Nome do pacote: só letras minúsculas e números, sem espaços nem hífens
package.name = adaptadorpedagogico

# (str) Domínio (ao contrário). ID final: org.educacaoinclusiva.adaptadorpedagogico
# Não o altere depois de instalar o app: o Android trataria como outro app.
package.domain = org.educacaoinclusiva

# (str) Pasta onde está o main.py
source.dir = .

# (list) Extensões de ficheiros incluídas no APK
source.include_exts = py,png

# (list) Pastas que não entram no APK
source.exclude_dirs = tests, bin, venv, .venv, .buildozer, .github, .git, __pycache__

# (str) Versão do app. No GitHub Actions é substituída automaticamente por
# 1.0.<número da compilação> (variável APP_VERSION), para que cada APK novo
# se instale por cima do anterior.
version = 1.0.0

# (list) Bibliotecas Python.
# - kivy 2.3.1 é a versão da receita do python-for-android 2026.05.09;
# - requests/urllib3/idna/certifi: HTTPS para a API da Anthropic (certifi sem
#   versão fixa para trazer sempre os certificados raiz mais recentes);
# - chardet 5.2.0: dependência que a receita do Kivy já pede; fixada porque
#   as versões 6+ trazem módulos compilados para PC que não servem no Android.
# Não acrescente charset-normalizer: o requests funciona com o chardet e as
# versões recentes do charset-normalizer trazem binários nativos.
requirements = python3,kivy==2.3.1,requests==2.34.2,urllib3==2.8.0,idna==3.20,certifi,chardet==5.2.0

# (str) Ecrã de abertura e ícones (ícone adaptativo usado no Android 8+)
presplash.filename = %(source.dir)s/data/presplash.png
icon.filename = %(source.dir)s/data/icon.png
icon.adaptive_foreground.filename = %(source.dir)s/data/icon_fg.png
icon.adaptive_background.filename = %(source.dir)s/data/icon_bg.png

# (list) Orientação: portrait = vertical
orientation = portrait

# (bool) 0 = mantém a barra de estado do Android visível
fullscreen = 0


# -----------------------------------------------------------------------------
#  Android
# -----------------------------------------------------------------------------

# (str) Cor de fundo do ecrã de abertura
android.presplash_color = #2F5DA8

# (list) Permissões: Internet para a API e estado da rede para avisar quando
# o aparelho está sem conexão.
android.permissions = android.permission.INTERNET, android.permission.ACCESS_NETWORK_STATE

# (int) API alvo (targetSdk/compileSdk). 33 é a recomendada pelo
# python-for-android para o bootstrap SDL2 e evita que o Android 15+ desenhe
# o app por baixo das barras do sistema (edge-to-edge obrigatório a partir do
# alvo 35). Para publicar na Play Store será preciso subir este valor.
android.api = 33

# (int) API mínima (Android 5.0). Mantenha android.minapi e android.ndk_api
# iguais. Não suba para 24+ com este python-for-android: desde agosto de 2026
# o charset-normalizer publica pacotes nativos para Android 24+ que esta versão
# do python-for-android não consegue instalar, e a compilação falha.
android.minapi = 21
android.ndk_api = 21

# (str) Versão do NDK (a recomendada pelo python-for-android 2026.05.09)
android.ndk = 28c

# (bool) Aceita as licenças do Android SDK automaticamente (necessário na nuvem)
android.accept_sdk_license = True

# (list) Arquiteturas: arm64-v8a (telemóveis atuais) e armeabi-v7a (aparelhos
# de entrada com Android 32 bits). Para compilar mais depressa em testes,
# deixe só arm64-v8a.
android.archs = arm64-v8a, armeabi-v7a

# (bool) Sem backup automático: a chave da API guardada no aparelho não vai
# para a nuvem nem para backups por cabo.
android.allow_backup = False

# (str) Formato gerado por "buildozer android debug"
android.debug_artifact = apk

# (str) Formato gerado por "buildozer android release" (Play Store usa aab)
android.release_artifact = aab

# (str) Filtro do logcat ao depurar com "buildozer android logcat"
android.logcat_filters = *:S python:D


# -----------------------------------------------------------------------------
#  python-for-android (p4a)
# -----------------------------------------------------------------------------

# (str) Ramo e commit fixos = versão 2026.05.09 (compilações reprodutíveis).
# Para atualizar no futuro, troque o commit pelo de uma versão mais recente.
p4a.branch = master
p4a.commit = 58d21141f17c889bf8585f5665921d72028f8831

# (str) Bootstrap: sdl2 (o Kivy 2.3.1 usa SDL2)
p4a.bootstrap = sdl2


[buildozer]

# (int) Nível de log: 2 = mostra a saída completa (útil para diagnosticar erros)
log_level = 2

# (int) Avisa se for executado como root
warn_on_root = 1
