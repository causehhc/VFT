<template>
  <div class="Base_Container">
    <el-container>
      <el-header class="Header_Wrapper">
        <Navbar/>
      </el-header>
      <el-main class="Main_Wrapper">
        <AppMain/>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import AppMain from "./components/AppMain";
import Navbar from "./components/Navbar";
export default {
  name: "index",
  components: {AppMain, Navbar}
}
</script>

<style scoped>
.Base_Container {

}
.Header_Wrapper{

}
.Main_Wrapper{
  height: 650px;
  /*height: 100%;*/
  /*position:relative;*/
  /*overflow-y: hidden;*/
  background: #eff2f2;
}

</style>
